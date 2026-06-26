// Simultaneous Translation — client.
// One surface, two input modes that drive the same translation:
//   • type  → POST /api/translate/step per settled word
//   • speak → mic → WebSocket /api/asr (faster-whisper on the server)
// Translation is served by a remote LLM when available, else the local model.
// A small mascot reacts to state; a Clear button kills any running process.

const state = { srcLang: 'en', tgtLang: 'te', k: 3, languages: [], examples: {}, backend: 'local' };
const DEFAULT_LANGS = [
    { code: 'te', name: 'Telugu', script: 'తెలుగు' },
    { code: 'hi', name: 'Hindi', script: 'हिन्दी' },
    { code: 'gu', name: 'Gujarati', script: 'ગુજરાતી' },
    { code: 'ta', name: 'Tamil', script: 'தமிழ்' },
    { code: 'en', name: 'English', script: 'English' },
];
const $ = (id) => document.getElementById(id);

document.addEventListener('DOMContentLoaded', async () => {
    $('translation-output').dataset.placeholder = 'Translation appears here, live.';
    setMicIcon(false);
    setMascot('idle');
    buildWaveform();
    await loadBackend();
    await loadLanguages();
    await loadExamples();
    wireControls();
    setStatus(`${state.backend} · wait-${state.k}`, false);
});

// ---------- Backend banner ----------
async function loadBackend() {
    const b = await fetchJSON('/api/backend');
    if (b?.backend) state.backend = b.backend;
}

// ---------- Languages ----------
async function loadLanguages() {
    state.languages = (await fetchJSON('/api/languages'))?.languages || DEFAULT_LANGS;
    fillSelect($('src-lang'), state.srcLang);
    fillSelect($('tgt-lang'), state.tgtLang);
    syncLabels();
}
function fillSelect(sel, current) {
    sel.innerHTML = '';
    for (const l of state.languages) {
        const o = document.createElement('option');
        o.value = l.code;
        o.textContent = l.name;
        o.selected = l.code === current;
        sel.appendChild(o);
    }
}
function langName(code) {
    return state.languages.find((l) => l.code === code)?.name || code;
}
function syncLabels() {
    $('src-label').textContent = langName(state.srcLang);
    $('tgt-label').textContent = langName(state.tgtLang);
    $('source-input').placeholder = `Type in ${langName(state.srcLang)}, or speak…`;
}

// ---------- Examples ----------
async function loadExamples() {
    state.examples = (await fetchJSON('/api/examples'))?.examples || {};
    renderExamples();
}
function renderExamples() {
    const bar = $('examples');
    bar.innerHTML = '';
    for (const text of (state.examples[state.srcLang] || []).slice(0, 3)) {
        const b = document.createElement('button');
        b.className = 'example';
        b.textContent = text;
        b.onclick = () => { $('source-input').value = text + ' '; restart(); $('source-input').focus(); };
        bar.appendChild(b);
    }
}

// ---------- Controls ----------
function wireControls() {
    $('src-lang').onchange = (e) => setLang('src', e.target.value);
    $('tgt-lang').onchange = (e) => setLang('tgt', e.target.value);
    $('swap').onclick = () => {
        [state.srcLang, state.tgtLang] = [state.tgtLang, state.srcLang];
        $('src-lang').value = state.srcLang;
        $('tgt-lang').value = state.tgtLang;
        afterLangChange();
    };
    const ks = $('k-slider');
    ks.oninput = () => {
        state.k = parseInt(ks.value);
        $('k-display').textContent = `wait-${state.k}`;
        setStatus(`${state.backend} · wait-${state.k}`, mic.active);
        if (mic.active) { sendMic({ type: 'config', k: state.k }); }
        else { clearTimeout(wk.timer); wk.timer = setTimeout(restart, 500); }
    };
    $('source-input').addEventListener('input', onType);
    $('mic-btn').onclick = toggleMic;
    $('clear-btn').onclick = killAll;
}
function setLang(which, code) {
    if (which === 'src') {
        if (code === state.tgtLang) { state.tgtLang = state.srcLang; $('tgt-lang').value = state.tgtLang; }
        state.srcLang = code;
    } else {
        if (code === state.srcLang) { state.srcLang = state.tgtLang; $('src-lang').value = state.srcLang; }
        state.tgtLang = code;
    }
    afterLangChange();
}
function afterLangChange() {
    syncLabels();
    renderExamples();
    if (mic.active) sendMic({ type: 'config', target_lang: state.tgtLang, source_lang: state.srcLang });
    else restart();
}

// ---------- Output / status / mascot helpers ----------
function setOutput(text, streaming) {
    const el = $('translation-output');
    el.textContent = text || '';
    el.classList.toggle('streaming', !!streaming);
}
function setReadout(text) { $('readout').textContent = text || ''; }
function setStatus(text, live) {
    const el = $('status');
    if (!el) return;
    el.textContent = text || '';
    el.classList.toggle('live', !!live);
}
function setMascot(stateName) {
    const m = $('mascot');
    if (!m) return;
    m.classList.remove('idle', 'listening', 'thinking', 'happy');
    m.classList.add(stateName);
}

// ---------- Clear / kill ----------
// One button to stop everything in flight and wipe the surface clean.
function killAll() {
    if (mic.active) stopMic();
    else teardownMic();
    clearTimeout(wk.timer);
    clearTimeout(mic.silenceTimer);
    resetWk();
    const inp = $('source-input');
    inp.value = '';
    inp.readOnly = false;
    setOutput('', false);
    setReadout('');
    setLevel(0);
    setStatus(`${state.backend} · wait-${state.k}`, false);
    setMascot('idle');
    inp.focus();
}

// =============================================================================
// Typed wait-k: each settled word costs one /step; a short pause flushes the
// tail, and a longer pause (the sentence is "done") triggers a full-sentence
// re-translation that corrects the word order wait-k guessed at low k.
// =============================================================================
const FINALIZE_MS = 1600;    // pause this long → flush the wait-k tail
const CORRECT_MS = 3000;     // pause this long → sentence done → full re-translate
let wk = { committed: [], read: 0, words: [], running: false,
           timer: null, correctTimer: null, corrected: false };

function settledWords() {
    const v = $('source-input').value;
    const parts = v.split(/\s+/).filter(Boolean);
    if (!/\s$/.test(v)) parts.pop();   // trailing word isn't settled until a space
    return parts;
}
function allWords() {
    return $('source-input').value.split(/\s+/).filter(Boolean);
}
function extendsRead(words) {
    if (words.length < wk.read) return false;
    for (let i = 0; i < wk.read; i++) if (words[i] !== wk.words[i]) return false;
    return true;
}

function onType() {
    if (mic.active) return;
    clearTimeout(wk.timer);
    if (!$('source-input').value.trim()) { resetWk(); setMascot('idle'); return; }
    setMascot('thinking');
    pump();
    wk.timer = setTimeout(finalize, FINALIZE_MS);
}

function resetWk() {
    wk.committed = []; wk.read = 0; wk.words = [];
    setOutput('', false);
    setReadout('');
}

function restart() {
    clearTimeout(wk.timer);
    resetWk();
    if ($('source-input').value.trim()) {
        setMascot('thinking');
        pump();
        wk.timer = setTimeout(finalize, FINALIZE_MS);
    }
}

async function pump() {
    if (wk.running) return;
    wk.running = true;
    try {
        while (true) {
            const words = settledWords();
            if (!extendsRead(words)) { resetWk(); continue; }
            if (words.length <= wk.read) break;
            await doStep(words, false);
        }
    } catch (_) { /* surfaced in readout */ } finally { wk.running = false; }
}

async function finalize() {
    if (wk.running) { wk.timer = setTimeout(finalize, 250); return; }
    const words = allWords();
    if (!words.length) return;
    wk.running = true;
    try {
        if (!extendsRead(words)) resetWk();
        await doStep(words, true);
        setMascot('happy');
    } catch (_) { /* surfaced in readout */ } finally { wk.running = false; }
}

async function doStep(words, finalizeFlag) {
    setOutput($('translation-output').textContent, !finalizeFlag);
    let res;
    try {
        res = await fetch('/api/translate/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                words, committed: wk.committed, read: wk.read,
                k: state.k, target_lang: state.tgtLang, finalize: finalizeFlag,
            }),
        });
    } catch (e) { setReadout('Server unreachable.'); throw e; }
    if (!res.ok) { setReadout(`Server error ${res.status}.`); throw new Error(res.status); }
    const d = await res.json();
    if (d.error) { setReadout(d.error); throw new Error(d.error); }

    wk.committed = d.committed;
    wk.read = d.read;
    wk.words = words.slice(0, d.read);
    setOutput(d.translation_so_far, !finalizeFlag);
    const written = d.committed.length || (d.translation_so_far || '').split(/\s+/).filter(Boolean).length;
    setReadout(`wait-${state.k} · ${wk.read} read · ${written} written`);
}

// =============================================================================
// Speech: mic → 16 kHz PCM over WebSocket → server ASR → live translation.
// The transcript fills the same source field; the translation streams as usual.
// Auto-stops after a short silence (treated as "sentence finished") so the full
// translation is flushed without waiting forever.
// =============================================================================
const ICON_MIC = '<span class="material-symbols-outlined">mic</span>';
const ICON_STOP = '<span class="material-symbols-outlined">stop</span>';

const SILENCE_MS = 2500;     // this much continuous silence = "sentence finished"
const VOICE_RMS = 0.012;     // frame RMS above this counts as speech

let mic = {
    ws: null, ctx: null, source: null, node: null, sink: null, stream: null,
    active: false, lastVoice: 0, silenceTimer: null,
};

function setMicIcon(recording) {
    const btn = $('mic-btn');
    btn.innerHTML = recording ? ICON_STOP : ICON_MIC;
    btn.classList.toggle('recording', recording);
    btn.title = recording ? 'Stop' : 'Speak';
}
function sendMic(obj) {
    if (mic.ws && mic.ws.readyState === WebSocket.OPEN) mic.ws.send(JSON.stringify(obj));
}

async function toggleMic() {
    if (mic.active) { stopMic(); return; }
    try { await startMic(); }
    catch (e) { setReadout(`Microphone unavailable — ${e.message || e}`); teardownMic(); setMicIcon(false); setMascot('idle'); }
}

async function startMic() {
    clearTimeout(wk.timer);
    resetWk();
    $('source-input').value = '';
    $('source-input').readOnly = true;

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    mic.ws = new WebSocket(`${proto}://${location.host}/api/asr`);
    mic.ws.binaryType = 'arraybuffer';
    mic.ws.onmessage = onAsrMessage;
    mic.ws.onclose = () => { if (mic.active) stopMic(); };
    await new Promise((resolve, reject) => {
        mic.ws.onopen = resolve;
        mic.ws.onerror = () => reject(new Error('ASR server unreachable'));
    });
    sendMic({ type: 'start', target_lang: state.tgtLang, source_lang: state.srcLang, k: state.k });

    mic.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    mic.ctx = new AudioContext({ sampleRate: 16000 });
    await mic.ctx.audioWorklet.addModule('/static/pcm-worklet.js');
    mic.source = mic.ctx.createMediaStreamSource(mic.stream);
    mic.node = new AudioWorkletNode(mic.ctx, 'pcm-processor');
    mic.node.port.onmessage = onAudioFrame;
    mic.sink = mic.ctx.createGain();         // silent sink keeps the graph pulling
    mic.sink.gain.value = 0;
    mic.source.connect(mic.node);
    mic.node.connect(mic.sink);
    mic.sink.connect(mic.ctx.destination);

    mic.active = true;
    mic.lastVoice = performance.now();
    armSilenceWatch();
    setMicIcon(true);
    setOutput('', true);
    setReadout('Listening…');
    setStatus(`${state.backend} · listening`, true);
    setMascot('listening');
}

// Each ~100 ms audio frame: forward PCM to the server, track loudness for the
// waveform and the silence auto-stop.
function onAudioFrame(e) {
    const { pcm, level } = e.data;
    if (mic.ws && mic.ws.readyState === WebSocket.OPEN) mic.ws.send(pcm);
    setLevel(level);
    if (level > VOICE_RMS) mic.lastVoice = performance.now();
}

// Poll for a short continuous silence → auto-stop (finalize the utterance).
function armSilenceWatch() {
    clearTimeout(mic.silenceTimer);
    const tick = () => {
        if (!mic.active) return;
        if (performance.now() - mic.lastVoice >= SILENCE_MS) { stopMic(); return; }
        mic.silenceTimer = setTimeout(tick, 250);
    };
    mic.silenceTimer = setTimeout(tick, 250);
}

function stopMic() {
    if (!mic.active) return;
    mic.active = false;
    clearTimeout(mic.silenceTimer);
    setMicIcon(false);
    setLevel(0);
    sendMic({ type: 'stop' });               // ask the server to flush the tail
    teardownAudio();
    $('source-input').readOnly = false;
    setReadout('Finishing…');
    setStatus(`${state.backend} · wait-${state.k}`, false);
    setMascot('thinking');
    // The socket is normally closed in onAsrMessage when the final (complete)
    // translation arrives. This is only a safety net if that never comes — keep it
    // long enough that a slow full-sentence decode is never cut off mid-way.
    setTimeout(() => { if (mic.ws) { try { mic.ws.close(); } catch (_) {} mic.ws = null; } }, 30000);
}

function onAsrMessage(ev) {
    let d;
    try { d = JSON.parse(ev.data); } catch (_) { return; }

    if (d.type === 'partial' || d.type === 'final') {
        if (d.transcript) $('source-input').value = d.transcript;
        // Transcript-only and translation-only partials interleave (the server
        // pushes the transcript first, then the translation): only touch the
        // field that's present so neither clobbers the other. A missing/null
        // `translation` (READ phase / no new stable word) keeps what's shown.
        if (d.translation !== undefined && d.translation !== null) setOutput(d.translation, d.type === 'partial');
        // Source language is pinned by the user's selection — never auto-switched
        // from ASR. Just reflect the active direction in the readout.
        setReadout(`wait-${state.k} · ${langName(state.srcLang)} → ${langName(state.tgtLang)}`);
        if (d.type === 'partial') setMascot('listening');
        if (d.type === 'final') {
            setOutput(d.translation, false);
            setMascot('happy');
            if (mic.ws) { try { mic.ws.close(); } catch (_) {} mic.ws = null; }
        }
    } else if (d.type === 'error') {
        setReadout(d.msg);
        setMascot('idle');
    }
}

function teardownAudio() {
    if (mic.node) { try { mic.node.disconnect(); } catch (_) {} mic.node = null; }
    if (mic.source) { try { mic.source.disconnect(); } catch (_) {} mic.source = null; }
    if (mic.sink) { try { mic.sink.disconnect(); } catch (_) {} mic.sink = null; }
    if (mic.stream) { mic.stream.getTracks().forEach((t) => t.stop()); mic.stream = null; }
    if (mic.ctx) { try { mic.ctx.close(); } catch (_) {} mic.ctx = null; }
}
function teardownMic() {
    mic.active = false;
    clearTimeout(mic.silenceTimer);
    teardownAudio();
    if (mic.ws) { try { mic.ws.close(); } catch (_) {} mic.ws = null; }
    $('source-input').readOnly = false;
    setLevel(0);
}

// ---------- Waveform ----------
let waveBars = [];
function buildWaveform() {
    const w = $('waveform');
    if (!w) return;
    w.innerHTML = '';
    waveBars = [];
    for (let i = 0; i < 9; i++) {
        const b = document.createElement('span');
        b.className = 'wbar';
        w.appendChild(b);
        waveBars.push(b);
    }
}
// Map frame loudness onto the bars with a little per-bar variation so it reads
// as a lively waveform. One cheap style write per bar per ~100 ms.
function setLevel(level) {
    if (!waveBars.length) return;
    const amp = Math.min(1, level * 9);     // normalise typical speech to ~0..1
    for (let i = 0; i < waveBars.length; i++) {
        const dist = Math.abs(i - (waveBars.length - 1) / 2) / waveBars.length;
        const h = amp <= 0.001 ? 0.06 : Math.max(0.06, amp * (1 - dist) * (0.7 + Math.random() * 0.6));
        waveBars[i].style.transform = `scaleY(${h.toFixed(3)})`;
    }
}

// ---------- util ----------
async function fetchJSON(url) {
    try { const r = await fetch(url); return await r.json(); } catch (_) { return null; }
}

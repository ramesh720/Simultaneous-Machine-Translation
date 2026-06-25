// =============================================================================
// Simultaneous MT — Frontend Application
// Live (as-you-type) wait-k translation, READ/WRITE trace visualization, and
// an on-demand full-sentence quality comparison.
// =============================================================================

// --- State ---
let state = {
    srcLang: 'te',
    tgtLang: 'en',
    k: 3,
    languages: [],
    examples: {},
    // Latest live translation (used by the quality check)
    lastText: null,
    lastWaitk: null,
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    await loadLanguages();
    await loadExamples();
    setupKSlider();
    setupSwapButton();
    setupLiveInput();
});

// --- Language Setup ---
async function loadLanguages() {
    try {
        const res = await fetch('/api/languages');
        const data = await res.json();
        state.languages = data.languages;
        renderLanguageSelectors();
    } catch (e) {
        console.warn('Could not load languages, using defaults');
        state.languages = [
            { code: 'te', name: 'Telugu', script: 'తెలుగు' },
            { code: 'hi', name: 'Hindi', script: 'हिन्दी' },
            { code: 'gu', name: 'Gujarati', script: 'ગુજરાતી' },
            { code: 'ta', name: 'Tamil', script: 'தமிழ்' },
            { code: 'en', name: 'English', script: 'English' },
        ];
        renderLanguageSelectors();
    }
}

function renderLanguageSelectors() {
    const srcContainer = document.getElementById('src-lang-selector');
    const tgtContainer = document.getElementById('tgt-lang-selector');
    srcContainer.innerHTML = '';
    tgtContainer.innerHTML = '';

    state.languages.forEach(lang => {
        // Source selector
        const srcBtn = document.createElement('button');
        srcBtn.className = `lang-btn ${lang.code === state.srcLang ? 'active' : ''}`;
        srcBtn.innerHTML = `${lang.name} <span class="script">${lang.script}</span>`;
        srcBtn.onclick = () => selectSrcLang(lang.code);
        srcContainer.appendChild(srcBtn);

        // Target selector
        const tgtBtn = document.createElement('button');
        tgtBtn.className = `lang-btn ${lang.code === state.tgtLang ? 'active' : ''}`;
        tgtBtn.innerHTML = `${lang.name} <span class="script">${lang.script}</span>`;
        tgtBtn.onclick = () => selectTgtLang(lang.code);
        tgtContainer.appendChild(tgtBtn);
    });

    updateLabels();
    renderExamples();
}

function selectSrcLang(code) {
    if (code === state.tgtLang) {
        state.tgtLang = state.srcLang;
    }
    state.srcLang = code;
    renderLanguageSelectors();
    restart();
}

function selectTgtLang(code) {
    if (code === state.srcLang) {
        state.srcLang = state.tgtLang;
    }
    state.tgtLang = code;
    renderLanguageSelectors();
    restart();
}

function updateLabels() {
    const srcLang = state.languages.find(l => l.code === state.srcLang);
    const tgtLang = state.languages.find(l => l.code === state.tgtLang);
    document.getElementById('input-label').textContent = `Source (${srcLang?.name || state.srcLang})`;
    document.getElementById('output-label').textContent = `Translation (${tgtLang?.name || state.tgtLang})`;
}

function setupSwapButton() {
    document.getElementById('swap-direction').addEventListener('click', () => {
        const tmp = state.srcLang;
        state.srcLang = state.tgtLang;
        state.tgtLang = tmp;
        renderLanguageSelectors();
        restart();
    });
}

// --- Examples ---
async function loadExamples() {
    try {
        const res = await fetch('/api/examples');
        const data = await res.json();
        state.examples = data.examples;
        renderExamples();
    } catch (e) {
        console.warn('Could not load examples');
        state.examples = {
            te: ['నేను రోజూ ఉదయం పార్కులో నడుస్తాను.', 'భారతదేశం ప్రపంచంలో అతి పెద్ద ప్రజాస్వామ్య దేశం.'],
            hi: ['मैं हर रोज सुबह पार्क में टहलता हूँ.', 'भारत दुनिया का सबसे बड़ा लोकतंत्र है.'],
            gu: ['હું દરરોજ સવારે પાર્કમાં ચાલું છું.', 'ભારત વિશ્વનું સૌથી મોટું લોકશાહી છે.'],
            ta: ['நான் தினமும் காலையில் பூங்காவில் நடப்பேன்.', 'இந்தியா உலகின் மிகப்பெரிய ஜனநாயக நாடு.'],
            en: ['I walk in the park every morning.', 'India is the largest democracy in the world.'],
        };
        renderExamples();
    }
}

function renderExamples() {
    const bar = document.getElementById('examples-bar');
    bar.innerHTML = '';
    const examples = state.examples[state.srcLang] || [];
    examples.forEach(text => {
        const chip = document.createElement('button');
        chip.className = 'example-chip';
        chip.textContent = text;
        chip.onclick = () => {
            // Trailing space so every word is "settled" and translates immediately.
            document.getElementById('source-input').value = text + ' ';
            restart();
        };
        bar.appendChild(chip);
    });
}

// --- K Slider (applied to the live translation in real time) ---
function setupKSlider() {
    const slider = document.getElementById('k-slider');
    const display = document.getElementById('k-display');
    slider.addEventListener('input', () => {
        state.k = parseInt(slider.value);
        display.textContent = `k=${state.k}`;
        // Debounce: dragging fires many events; re-translate once it settles.
        clearTimeout(wk.typingTimer);
        wk.typingTimer = setTimeout(restart, 600);
    });
}

// --- Status ---
function setStatus(text, type = '') {
    const bar = document.getElementById('status-bar');
    const textEl = document.getElementById('status-text');
    bar.className = `status-bar ${type}`;
    textEl.textContent = text;
}

// --- Pipeline Animation ---
function activatePipelineStep(stepId) {
    document.querySelectorAll('.pipeline-step').forEach(s => s.classList.remove('active'));
    if (stepId) {
        document.getElementById(stepId)?.classList.add('active');
    }
}

// =============================================================================
// Incremental wait-k translation.
// On each completed word the client posts /api/translate/step, which advances
// the translation by ~one forward pass (re-using the tokens committed so far).
// After 2s of paused typing we "finalize" (read the rest + flush to EOS) and
// reveal the full-sentence quality button.
// =============================================================================
const FINALIZE_MS = 2000;

// Decode state, kept in sync with the server step by step.
let wk = {
    committed: [],   // target token ids emitted so far
    read: 0,         // source words already read by the model
    words: [],       // the source words already read (for edit detection)
    running: false,  // a /step request is in flight (serializes steps)
    typingTimer: null,
};

function setupLiveInput() {
    document.getElementById('source-input').addEventListener('input', onSourceInput);
}

// Completed words: trailing word is excluded until a space follows it, so we
// translate once per finished word.
function settledWords() {
    const v = document.getElementById('source-input').value;
    const parts = v.split(/\s+/).filter(Boolean);
    if (!/\s$/.test(v)) parts.pop();
    return parts;
}
function allWords() {
    return document.getElementById('source-input').value.split(/\s+/).filter(Boolean);
}

function onSourceInput() {
    clearTimeout(wk.typingTimer);
    hideQualityButton();
    if (!document.getElementById('source-input').value.trim()) {
        resetWk();
        setStatus('Ready', '');
        return;
    }
    pump();                                              // translate completed words
    wk.typingTimer = setTimeout(finalize, FINALIZE_MS);  // flush + button on pause
}

function resetWk() {
    wk.committed = []; wk.read = 0; wk.words = [];
    document.getElementById('translation-output').innerHTML =
        '<span class="placeholder-text" style="color: var(--text-muted)">Translation will appear here as you type...</span>';
    document.getElementById('trace-container').innerHTML =
        '<div style="color: var(--text-muted); font-style: italic;">READ / WRITE steps will appear here as you type…</div>';
    document.getElementById('trace-stats').textContent = '';
    activatePipelineStep(null);
}

// Restart the translation for the current text (after k / language change).
function restart() {
    clearTimeout(wk.typingTimer);
    resetWk();
    hideQualityButton();
    if (document.getElementById('source-input').value.trim()) {
        pump();
        wk.typingTimer = setTimeout(finalize, FINALIZE_MS);
    } else {
        setStatus('Ready', '');
    }
}

// Does `words` extend the words already read (normal typing) vs an edit?
function extendsRead(words) {
    if (words.length < wk.read) return false;
    for (let i = 0; i < wk.read; i++) if (words[i] !== wk.words[i]) return false;
    return true;
}

// Translate any settled words not yet read. Serialized via wk.running.
async function pump() {
    if (wk.running) return;
    wk.running = true;
    try {
        while (true) {
            const words = settledWords();
            if (!extendsRead(words)) { resetWk(); continue; }  // edited → restart
            if (words.length <= wk.read) break;                 // nothing new
            await doStep(words, false);
        }
    } catch (_) {
        // doStep already reported the error to the status bar.
    } finally {
        wk.running = false;
    }
}

// On a typing pause: read remaining words (incl. the in-progress one), flush the
// tail to EOS, then reveal the quality button.
async function finalize() {
    if (wk.running) { wk.typingTimer = setTimeout(finalize, 300); return; }
    const words = allWords();
    if (words.length === 0) return;
    wk.running = true;
    try {
        if (!extendsRead(words)) resetWk();
        await doStep(words, true);
        if (wk.committed.length) {
            setStatus(`Done — wait-${state.k} complete (${words.length} words → ${wk.committed.length} tokens)`, 'active');
            activatePipelineStep('pipe-target');
            showQualityButton();
        }
    } catch (_) {
        // reported in doStep
    } finally {
        wk.running = false;
    }
}

// One server step: read new words, render new READ/WRITE events.
async function doStep(words, finalizeFlag) {
    const outputEl = document.getElementById('translation-output');
    const traceEl = document.getElementById('trace-container');
    const statsEl = document.getElementById('trace-stats');

    setStatus(finalizeFlag ? `Finishing (wait-${state.k})…` : `Translating (wait-${state.k})…`, 'streaming');

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
    } catch (e) {
        setStatus('Error: could not reach the server. Is server.py running (and restarted)?', '');
        throw e;
    }
    if (!res.ok) {
        setStatus(`Error: server returned ${res.status}. Restart server.py after code changes.`, '');
        throw new Error(`HTTP ${res.status}`);
    }
    const d = await res.json();
    if (d.error) { setStatus(`Error: ${d.error}`, ''); throw new Error(d.error); }

    // Drop the trace placeholder once real steps arrive.
    if (!traceEl.querySelector('.trace-step') && (d.reads.length || d.writes.length)) {
        traceEl.innerHTML = '';
    }

    let r = wk.read;
    d.reads.forEach(word => {
        r++;
        activatePipelineStep('pipe-read');
        addTraceStep(traceEl, 'read', `"${word}" (${r}/${d.src_total})`);
    });
    d.writes.forEach(w => {
        activatePipelineStep('pipe-write');
        addTraceStep(traceEl, 'write', `"${w.token}" → ${w.translation_so_far}`);
    });

    wk.committed = d.committed;
    wk.read = d.read;
    wk.words = words.slice(0, d.read);

    if (d.translation_so_far) outputEl.textContent = d.translation_so_far;
    statsEl.textContent = `READ: ${wk.read} | WRITE: ${wk.committed.length}`;

    if (!finalizeFlag) {
        setStatus(d.writes.length
            ? `Translating (wait-${state.k}) — ${wk.read} words read`
            : `Reading… (wait-${state.k} lag)`, d.writes.length ? 'streaming' : 'active');
    }

    // Keep latest output available for the quality check.
    state.lastText = document.getElementById('source-input').value.trim();
    state.lastWaitk = d.translation_so_far;
}

function addTraceStep(container, action, detail) {
    const step = document.createElement('div');
    step.className = 'trace-step';
    step.innerHTML = `
        <span class="trace-action ${action}">${action}</span>
        <span class="trace-detail">${escapeHtml(detail)}</span>
    `;
    container.appendChild(step);
    container.scrollTop = container.scrollHeight;
}

// --- Quality button visibility ---
function showQualityButton() {
    document.getElementById('quality-action').style.display = '';
}
function hideQualityButton() {
    document.getElementById('quality-action').style.display = 'none';
}

// =============================================================================
// Full-sentence check: offline translation + COMET / AL / LAAL for the wait-k run
// =============================================================================
async function checkQuality() {
    if (!state.lastText || !state.lastWaitk) return;

    const btn = document.getElementById('btn-check-quality');
    const results = document.getElementById('quality-results');
    btn.disabled = true;
    btn.textContent = '⏳ Translating full sentence + scoring…';
    results.innerHTML = '<div class="card"><span class="loading-shimmer" style="display:inline-block;width:70%;height:1.2em;border-radius:4px;">&nbsp;</span>'
        + '<p class="section-subtitle" style="margin-top:12px">Running the offline translation and COMET (COMET runs on CPU — first use also downloads the model, so this can take a while).</p></div>';

    try {
        const res = await fetch('/api/translate/quality', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: state.lastText,
                target_lang: state.tgtLang,
                waitk_translation: state.lastWaitk,
                tgt_tokens: wk.committed.length,
                k: state.k,
            }),
        });
        if (!res.ok) throw new Error(`server returned ${res.status}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderQualityResults(data);
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
        results.innerHTML = `<div class="card"><span style="color:#f43f5e">Error: ${escapeHtml(e.message)}</span></div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '📊 Check full sentence';
    }
}

function renderQualityResults(data) {
    const fmt = v => (v === null || v === undefined) ? 'N/A' : v;
    const cometCard = (data.comet !== null && data.comet !== undefined)
        ? `<div class="metric-value">${data.comet}</div>
           <div class="metric-label">COMET (Wait-K vs offline)</div>`
        : `<div class="metric-value" style="font-size:0.95rem">N/A</div>
           <div class="metric-label">${escapeHtml(data.comet_error || 'COMET unavailable')}</div>`;
    const results = document.getElementById('quality-results');
    results.innerHTML = `
        <div class="metrics-grid" style="margin-bottom:24px">
            <div class="metric-card">${cometCard}</div>
            <div class="metric-card">
                <div class="metric-value">${fmt(data.al)}</div>
                <div class="metric-label">Average Lagging (AL)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${fmt(data.laal)}</div>
                <div class="metric-label">Length-Adaptive AL</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">wait-${fmt(data.k)}</div>
                <div class="metric-label">Policy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${fmt(data.src_words)}</div>
                <div class="metric-label">Source words</div>
            </div>
        </div>
        <div class="comparison-grid">
            <div class="card comparison-card simultaneous">
                <span class="card-badge">Simultaneous</span>
                <h3 style="margin:8px 0">Wait-K (live)</h3>
                <div class="output-box">${escapeHtml(state.lastWaitk)}</div>
            </div>
            <div class="card comparison-card offline">
                <span class="card-badge">Offline</span>
                <h3 style="margin:8px 0">Full sentence (reference)</h3>
                <div class="output-box">${escapeHtml(data.full_translation)}</div>
            </div>
        </div>
    `;
}

// --- Clear ---
function clearAll() {
    clearTimeout(wk.typingTimer);
    document.getElementById('source-input').value = '';
    resetWk();
    hideQualityButton();
    state.lastText = state.lastWaitk = null;
    document.getElementById('quality-results').innerHTML =
        '<div class="card"><span class="placeholder-text" style="color: var(--text-muted)">Type a sentence above, then click “Check full sentence” to compare the live Wait-K output against the offline translation (COMET, AL, LAAL).</span></div>';
    setStatus('Ready', '');
}

// --- Utilities ---
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

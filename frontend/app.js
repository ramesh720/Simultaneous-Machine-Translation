// =============================================================================
// Simultaneous MT — Frontend Application
// Handles language selection, streaming translation, trace visualization,
// policy comparison, and metrics charts.
// =============================================================================

// --- State ---
let state = {
    srcLang: 'te',
    tgtLang: 'en',
    k: 3,
    languages: [],
    examples: {},
    isTranslating: false,
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    await loadLanguages();
    await loadExamples();
    setupKSlider();
    setupSwapButton();
    loadMetrics();
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
}

function selectTgtLang(code) {
    if (code === state.srcLang) {
        state.srcLang = state.tgtLang;
    }
    state.tgtLang = code;
    renderLanguageSelectors();
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
            document.getElementById('source-input').value = text;
        };
        bar.appendChild(chip);
    });
}

// --- K Slider ---
function setupKSlider() {
    const slider = document.getElementById('k-slider');
    const display = document.getElementById('k-display');
    slider.addEventListener('input', () => {
        state.k = parseInt(slider.value);
        display.textContent = `k=${state.k}`;
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

// --- Translation: Full Sentence ---
async function translateFull() {
    const text = document.getElementById('source-input').value.trim();
    if (!text || state.isTranslating) return;

    state.isTranslating = true;
    setStatus('Translating (full sentence)...', 'active');
    activatePipelineStep('pipe-source');

    const outputEl = document.getElementById('translation-output');
    outputEl.innerHTML = '<span class="loading-shimmer" style="display:inline-block;width:60%;height:1.2em;border-radius:4px;">&nbsp;</span>';

    try {
        const res = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, target_lang: state.tgtLang }),
        });
        const data = await res.json();
        outputEl.textContent = data.translation;
        activatePipelineStep('pipe-target');
        setStatus(`Done — Full sentence translation`, 'active');
    } catch (e) {
        outputEl.textContent = `Error: ${e.message}`;
        setStatus('Error', '');
    }

    state.isTranslating = false;
}

// --- Translation: Streaming Wait-K ---
let _eventSource = null;

function translateStream() {
    const text = document.getElementById('source-input').value.trim();
    if (!text || state.isTranslating) return;

    // Close any previous stream
    if (_eventSource) { _eventSource.close(); _eventSource = null; }

    state.isTranslating = true;
    setStatus(`Translating (wait-${state.k})...`, 'streaming');

    const outputEl = document.getElementById('translation-output');
    const traceEl = document.getElementById('trace-container');
    const statsEl = document.getElementById('trace-stats');

    outputEl.innerHTML = '<span class="cursor-blink"></span>';
    traceEl.innerHTML = '';
    statsEl.textContent = '';

    let readCount = 0, writeCount = 0;

    const params = new URLSearchParams({ text, target_lang: state.tgtLang, k: state.k });
    _eventSource = new EventSource(`/api/translate/stream?${params}`);

    _eventSource.addEventListener('read', (e) => {
        const data = JSON.parse(e.data);
        handleStreamEvent('read', data, outputEl, traceEl, statsEl);
        readCount++;
        statsEl.textContent = `READ: ${readCount} | WRITE: ${writeCount}`;
    });

    _eventSource.addEventListener('write', (e) => {
        const data = JSON.parse(e.data);
        handleStreamEvent('write', data, outputEl, traceEl, statsEl);
        writeCount++;
        statsEl.textContent = `READ: ${readCount} | WRITE: ${writeCount}`;
    });

    _eventSource.addEventListener('stop', (e) => {
        handleStreamEvent('stop', {}, outputEl, traceEl, statsEl);
    });

    _eventSource.addEventListener('done', (e) => {
        const data = JSON.parse(e.data);
        handleStreamEvent('done', data, outputEl, traceEl, statsEl);
        _eventSource.close(); _eventSource = null;
        state.isTranslating = false;
    });

    _eventSource.onerror = () => {
        outputEl.textContent = 'Error: could not stream translation. Check server logs.';
        setStatus('Error', '');
        _eventSource.close(); _eventSource = null;
        state.isTranslating = false;
    };
}

function handleStreamEvent(event, data, outputEl, traceEl, statsEl) {
    switch (event) {
        case 'read':
            activatePipelineStep('pipe-read');
            addTraceStep(traceEl, 'read', `"${data.word}" (${data.src_read}/${data.src_total})`);
            break;

        case 'write':
            activatePipelineStep('pipe-write');
            outputEl.innerHTML = escapeHtml(data.translation_so_far) +
                '<span class="cursor-blink"></span>';
            addTraceStep(traceEl, 'write', `"${data.token}" → ${data.translation_so_far}`);
            break;

        case 'stop':
            activatePipelineStep('pipe-target');
            addTraceStep(traceEl, 'stop', 'End of translation');
            // Remove cursor
            const cursor = outputEl.querySelector('.cursor-blink');
            if (cursor) cursor.remove();
            setStatus(`Done — Wait-${state.k} translation complete`, 'active');
            break;

        case 'done':
            activatePipelineStep('pipe-target');
            outputEl.textContent = data.translation;
            setStatus(`Done — Wait-${state.k} | ${data.src_words} src words → ${data.tgt_tokens} tokens`, 'active');
            break;
    }
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

// --- Comparison ---
async function runComparison() {
    const text = document.getElementById('source-input').value.trim();
    if (!text || state.isTranslating) return;

    state.isTranslating = true;
    const fullEl = document.getElementById('comparison-full');
    const waitkEl = document.getElementById('comparison-waitk');
    const fullStatus = document.getElementById('comparison-full-status');
    const waitkStatus = document.getElementById('comparison-waitk-status');

    fullEl.innerHTML = '<span class="loading-shimmer" style="display:inline-block;width:80%;height:1.2em;border-radius:4px;">&nbsp;</span>';
    waitkEl.innerHTML = '<span class="loading-shimmer" style="display:inline-block;width:80%;height:1.2em;border-radius:4px;">&nbsp;</span>';
    fullStatus.innerHTML = '<span class="status-dot"></span><span>Running both policies...</span>';
    fullStatus.className = 'status-bar streaming';
    waitkStatus.innerHTML = '<span class="status-dot"></span><span>Running both policies...</span>';
    waitkStatus.className = 'status-bar streaming';

    try {
        const res = await fetch('/api/translate/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, target_lang: state.tgtLang, k: state.k }),
        });
        const data = await res.json();

        // Full-sentence result
        fullEl.textContent = data.full_translation;
        fullStatus.innerHTML = '<span class="status-dot"></span><span>Complete (reads all → translates)</span>';
        fullStatus.className = 'status-bar active';

        // Wait-K result with metrics
        const m = data.metrics;
        waitkEl.innerHTML = `
            <div style="margin-bottom:12px">${escapeHtml(data.waitk_translation)}</div>
            <div style="padding:12px; background:rgba(0,0,0,0.25); border-radius:8px; border:1px solid var(--border-glass)">
                <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px">Latency Metrics</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.9rem">
                    <div>Average Lagging (AL):<br><strong style="color:var(--accent-primary); font-size:1.1rem">${m.al !== null ? m.al + ' words' : 'N/A'}</strong></div>
                    <div>Length-Adaptive AL:<br><strong style="color:var(--accent-primary); font-size:1.1rem">${m.laal !== null ? m.laal + ' words' : 'N/A'}</strong></div>
                </div>
            </div>
        `;
        waitkStatus.innerHTML = `<span class="status-dot"></span><span>Complete (Wait-${m.k}, ${m.src_words} source words)</span>`;
        waitkStatus.className = 'status-bar active';
    } catch (e) {
        fullEl.textContent = `Error: ${e.message}`;
        waitkEl.textContent = `Error: ${e.message}`;
        fullStatus.className = waitkStatus.className = 'status-bar';
    }

    state.isTranslating = false;
}

// --- Clear ---
function clearAll() {
    document.getElementById('source-input').value = '';
    document.getElementById('translation-output').innerHTML =
        '<span style="color: var(--text-muted)">Translation will appear here...</span>';
    document.getElementById('trace-container').innerHTML =
        '<div style="color: var(--text-muted); font-style: italic;">Trace will appear here when you run a Wait-K translation...</div>';
    document.getElementById('trace-stats').textContent = '';
    setStatus('Ready', '');
    activatePipelineStep(null);
}

// --- Metrics Dashboard ---
async function loadMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        if (data.results && data.results.length > 0) {
            renderMetricsSummary(data.results);
            renderBLEUvsALChart(data.results);
            renderBLEUbyLangChart(data.results);
        } else {
            document.getElementById('metrics-summary').innerHTML =
                '<div class="metric-card" style="grid-column: 1/-1;"><p style="color:var(--text-muted)">No evaluation results yet. Run the evaluation pipeline first.</p></div>';
        }
    } catch (e) {
        console.warn('Could not load metrics:', e);
    }
}

function renderMetricsSummary(results) {
    const container = document.getElementById('metrics-summary');
    const fullResults = results.filter(r => r.policy === 'full');
    const waitkResults = results.filter(r => r.policy !== 'full');

    const avgBleu = results.reduce((s, r) => s + (r.bleu || 0), 0) / results.length;
    const avgAL = waitkResults.filter(r => r.AL).reduce((s, r) => s + r.AL, 0) / (waitkResults.filter(r => r.AL).length || 1);
    const avgComet = results.filter(r => r.comet).reduce((s, r) => s + r.comet, 0) / (results.filter(r => r.comet).length || 1);
    const avgLAAL = waitkResults.filter(r => r.LAAL).reduce((s, r) => s + r.LAAL, 0) / (waitkResults.filter(r => r.LAAL).length || 1);

    container.innerHTML = `
        <div class="metric-card">
            <div class="metric-value">${avgBleu.toFixed(1)}</div>
            <div class="metric-label">Avg BLEU</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${avgComet > 0 ? avgComet.toFixed(3) : 'N/A'}</div>
            <div class="metric-label">Avg COMET</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${avgAL > 0 ? avgAL.toFixed(1) : 'N/A'}</div>
            <div class="metric-label">Avg AL</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${avgLAAL > 0 ? avgLAAL.toFixed(1) : 'N/A'}</div>
            <div class="metric-label">Avg LAAL</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${results.length}</div>
            <div class="metric-label">Evaluations</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${new Set(results.map(r => r.lang)).size}</div>
            <div class="metric-label">Languages</div>
        </div>
    `;
}

function renderBLEUvsALChart(results) {
    const canvas = document.getElementById('bleu-al-chart');
    const waitkResults = results.filter(r => r.AL != null && r.bleu != null);

    // Group by language
    const langs = [...new Set(waitkResults.map(r => r.lang))];
    const colors = { te: '#6366f1', hi: '#22c55e', gu: '#f59e0b', ta: '#f43f5e' };
    const langNames = { te: 'Telugu', hi: 'Hindi', gu: 'Gujarati', ta: 'Tamil' };

    const datasets = langs.map(lang => {
        const langData = waitkResults
            .filter(r => r.lang === lang)
            .sort((a, b) => a.AL - b.AL);
        return {
            label: langNames[lang] || lang,
            data: langData.map(r => ({ x: r.AL, y: r.bleu })),
            borderColor: colors[lang] || '#888',
            backgroundColor: (colors[lang] || '#888') + '33',
            pointRadius: 5,
            pointHoverRadius: 8,
            showLine: true,
            tension: 0.3,
        };
    });

    new Chart(canvas, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: BLEU=${ctx.parsed.y.toFixed(1)}, AL=${ctx.parsed.x.toFixed(1)}`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Average Lagging (AL)', color: '#94a3b8' },
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    title: { display: true, text: 'BLEU', color: '#94a3b8' },
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
            },
        },
    });
}

function renderBLEUbyLangChart(results) {
    const canvas = document.getElementById('bleu-lang-chart');
    const policies = [...new Set(results.map(r => r.policy))].sort();
    const langs = [...new Set(results.map(r => r.lang))];
    const langNames = { te: 'Telugu', hi: 'Hindi', gu: 'Gujarati', ta: 'Tamil' };
    const policyColors = {
        'full': '#6366f1',
        'wait-1': '#f43f5e',
        'wait-3': '#f59e0b',
        'wait-5': '#22c55e',
        'wait-7': '#3b82f6',
    };

    const datasets = policies.map(policy => ({
        label: policy,
        data: langs.map(lang => {
            const r = results.find(r => r.policy === policy && r.lang === lang);
            return r ? r.bleu : 0;
        }),
        backgroundColor: (policyColors[policy] || '#888') + '99',
        borderColor: policyColors[policy] || '#888',
        borderWidth: 1,
    }));

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: langs.map(l => langNames[l] || l),
            datasets,
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
            },
            scales: {
                x: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    title: { display: true, text: 'BLEU', color: '#94a3b8' },
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
            },
        },
    });
}

// --- Utilities ---
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

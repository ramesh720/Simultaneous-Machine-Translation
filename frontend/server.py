"""FastAPI backend for the Simultaneous MT presentation frontend.

Incremental wait-k decoding: the frontend posts one /step per word as the user
types, and the server advances the translation by re-using the tokens committed
so far (so each word costs ~one forward pass). A /quality endpoint compares the
simultaneous output against a full-sentence offline translation. Port 8080.

Usage:
    python server.py
    python server.py --adapter checkpoints_multilang/final --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import torch

# Add the waitk_finetune source to path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "waitk_finetune"))
from src.load import load_model  # noqa: E402
from src.waitk import (  # noqa: E402
    average_lagging,
    build_prompt,
    laal,
    stop_token_ids,
)

# ---------- Globals ----------
MODEL = None
TOKENIZER = None
DEVICE = None

# faster-whisper ASR. In remote-LLM mode the GPU is free, so this runs on CUDA
# (float16) for blazing-fast transcription; in local-MT mode it falls back to CPU.
WHISPER = None
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

# ---------- Translation backend ----------
# Remote OpenAI-compatible LLM is the primary translation engine; the local 4B
# model is an automatic fallback (lazy-loaded only if the remote is unreachable).
TRANSLATE_BACKEND = "auto"          # auto | remote | local
API_URL = "http://64.247.196.173:8080/v1"
API_KEY = "EMPTY"
API_MODEL = None                    # auto-discovered from /v1/models at startup
USE_REMOTE = False                  # decided in main() by _probe_remote()
REMOTE_OK = False                   # flips False if the remote fails at runtime
_HTTP = None                        # lazy httpx.AsyncClient
_LOCAL_ARGS = {}                    # captured in main() for lazy fallback load

LANG_MAP = {
    "te": "Telugu",
    "hi": "Hindi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "en": "English",
}

app = FastAPI(title="Simultaneous MT Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=str(HERE)), name="static")


# =============================================================================
# Translation backend: remote OpenAI-compatible LLM (primary) + local fallback.
# =============================================================================
def _http_client() -> httpx.AsyncClient:
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    return _HTTP


def _probe_remote() -> bool:
    """Startup check: confirm the remote LLM answers and discover its model id.
    Returns True if it can be used for translation."""
    global API_MODEL
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        with httpx.Client(timeout=httpx.Timeout(8.0, connect=5.0)) as c:
            if API_MODEL is None:                      # discover served model id
                try:
                    m = c.get(f"{API_URL}/models", headers=headers)
                    if m.status_code == 200:
                        items = m.json().get("data", [])
                        if items:
                            API_MODEL = items[0].get("id")
                except Exception:
                    pass
            ping = c.post(
                f"{API_URL}/chat/completions", headers=headers,
                json={"model": API_MODEL or "default", "max_tokens": 1,
                      "temperature": 0,
                      "messages": [{"role": "user", "content": "Hi"}]},
            )
            ping.raise_for_status()
            return True
    except Exception as exc:  # noqa: BLE001 — probe is best-effort
        print(f"[server] Remote LLM probe failed ({type(exc).__name__}: {exc}).")
        return False


async def _remote_translate(text: str, target_language: str,
                            max_tokens: int = 256) -> str:
    """Translate ``text`` via the remote OpenAI-compatible chat endpoint.

    Mirrors src.waitk.build_prompt's instruction so output matches the local
    model. Raises on any transport/HTTP error so the caller can fall back."""
    payload = {
        "model": API_MODEL or "default",
        "messages": [
            {"role": "system",
             "content": f"Translate the text below to {target_language}. "
                        "Output only the translation, with no quotes or notes."},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = await _http_client().post(
        f"{API_URL}/chat/completions", json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def _ensure_local():
    """Lazy-load the local 4B model the first time the remote fails. No-op if
    it is already loaded. This is the demo's safety net."""
    global MODEL, TOKENIZER
    if MODEL is not None:
        return
    print("[server] Loading LOCAL fallback model (remote unavailable)...")
    MODEL, TOKENIZER = load_model(
        _LOCAL_ARGS.get("base", "sarvamai/sarvam-translate"),
        _LOCAL_ARGS.get("adapter"), device=DEVICE,
        quantize_4bit=_LOCAL_ARGS.get("use_4bit", True),
        attn_implementation="eager",
    )
    print("[server] Local fallback model ready.")


async def _translate_full(text: str, target_language: str) -> str:
    """Full-string translation, remote-first with automatic local fallback.

    Used for the remote wait-k path (prefix re-translation) and offline
    translation. Returns '' for empty input."""
    global REMOTE_OK
    text = (text or "").strip()
    if not text:
        return ""
    if USE_REMOTE:
        try:
            return await _remote_translate(text, target_language)
        except Exception as exc:  # noqa: BLE001 — degrade to local, don't crash
            print(f"[server] Remote translate failed → local fallback "
                  f"({type(exc).__name__}: {exc}).")
            REMOTE_OK = False
            _ensure_local()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _offline_translate, text, target_language)


async def _step_remote(words: list[str], tgt_lang: str, k: int,
                       finalize: bool) -> dict:
    """Wait-k via prefix re-translation for the remote (black-box) backend.

    The whole source prefix read so far is translated each step; the wait-k
    lag knob is honoured by withholding output until ``k`` words are available.
    Returns the same JSON shape as the local /step so the client is unchanged."""
    n = len(words)
    if not finalize and n < k:
        out = ""                                       # wait-k lag (READ phase)
    else:
        out = await _translate_full(" ".join(words), tgt_lang)
    return {
        "committed": [], "read": n, "reads": [], "writes": [],
        "translation_so_far": out, "src_total": n, "done": finalize,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "index.html").read_text()


@app.get("/api/backend")
async def backend_info():
    """Which translation engine is live (so the UI can show it)."""
    return {"backend": "remote" if (USE_REMOTE and REMOTE_OK) else "local",
            "model": API_MODEL if (USE_REMOTE and REMOTE_OK) else "sarvam-translate",
            "whisper": {"model": WHISPER_MODEL, "device": WHISPER_DEVICE}}


@app.get("/api/languages")
async def languages():
    return {
        "languages": [
            {"code": "te", "name": "Telugu", "script": "తెలుగు"},
            {"code": "hi", "name": "Hindi", "script": "हिन्दी"},
            {"code": "gu", "name": "Gujarati", "script": "ગુજરાતી"},
            {"code": "ta", "name": "Tamil", "script": "தமிழ்"},
            {"code": "en", "name": "English", "script": "English"},
        ],
        "directions": [
            {"from": "te", "to": "en", "label": "Telugu → English"},
            {"from": "hi", "to": "en", "label": "Hindi → English"},
            {"from": "gu", "to": "en", "label": "Gujarati → English"},
            {"from": "ta", "to": "en", "label": "Tamil → English"},
            {"from": "en", "to": "te", "label": "English → Telugu"},
            {"from": "en", "to": "hi", "label": "English → Hindi"},
            {"from": "en", "to": "gu", "label": "English → Gujarati"},
            {"from": "en", "to": "ta", "label": "English → Tamil"},
        ],
    }


def _offline_translate(text: str, tgt_lang: str) -> str:
    """Full-sentence (offline) greedy translation of ``text`` into ``tgt_lang``."""
    _ensure_local()                       # remote mode: load the model on demand
    model_device = next(MODEL.parameters()).device
    prompt = build_prompt(TOKENIZER, text, tgt_lang)
    enc = TOKENIZER(prompt, return_tensors="pt").to(model_device)
    with torch.inference_mode():
        gen = MODEL.generate(
            **enc, max_new_tokens=256, do_sample=False,
            num_beams=1, pad_token_id=TOKENIZER.pad_token_id,
        )
    out = TOKENIZER.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out.split("<end_of_turn>")[0].strip()


@torch.inference_mode()
def _next_token(words_read: list[str], committed: list[int], tgt_lang: str,
                eos_ids, suppress_eos: bool) -> int:
    """One greedy decoding step for the current source prefix + committed tokens."""
    device = next(MODEL.parameters()).device
    prompt = build_prompt(TOKENIZER, " ".join(words_read), tgt_lang)
    ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(device)
    if committed:
        tail = torch.tensor([committed], device=device, dtype=ids.dtype)
        ids = torch.cat([ids, tail], dim=1)
    logits = MODEL(input_ids=ids).logits[0, -1]
    if suppress_eos:
        for eid in eos_ids:
            logits[eid] = float("-inf")
    return int(logits.argmax())


@torch.inference_mode()
def _generate_tail(words_read: list[str], committed: list[int], tgt_lang: str,
                   eos_ids, max_new: int = 200) -> list[int]:
    """Fast KV-cached completion of the translation, continuing from ``committed``
    and allowing EOS. Used to flush the rest of the sentence on finalize."""
    device = next(MODEL.parameters()).device
    prompt = build_prompt(TOKENIZER, " ".join(words_read), tgt_lang)
    ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(device)
    if committed:
        tail = torch.tensor([committed], device=device, dtype=ids.dtype)
        ids = torch.cat([ids, tail], dim=1)
    gen = MODEL.generate(
        input_ids=ids, max_new_tokens=max_new, do_sample=False,
        num_beams=1, pad_token_id=TOKENIZER.pad_token_id,
    )
    out = []
    for t in gen[0, ids.shape[1]:].tolist():
        if t in eos_ids:
            break
        out.append(t)
    return out


@app.post("/api/translate/step")
async def translate_step(request: Request):
    """Advance the wait-k translation by the words newly available since the
    last call. Stateless: the client passes back the tokens committed so far.

    Body: {words: [str], committed: [int], read: int, k, target_lang, finalize}
    - During typing (finalize=false) EOS is suppressed and we emit at most
      ``read - k + 1`` tokens (the wait-k schedule: write 1 token per word read
      after the initial k-word lag).
    - On finalize (typing paused) all remaining words are read and the tail is
      generated until EOS, completing the sentence.
    """
    body = await request.json()
    words = [w for w in body.get("words", []) if w]
    tgt_lang = LANG_MAP.get(body.get("target_lang", "en"), "English")
    k = max(1, int(body.get("k", 3)))
    finalize = bool(body.get("finalize", False))

    # Remote backend: stateless prefix re-translation (no token-level decode).
    if USE_REMOTE and REMOTE_OK:
        return await _step_remote(words, tgt_lang, k, finalize)

    # ----- Local backend (unchanged token-level wait-k) -----
    committed = [int(x) for x in body.get("committed", [])]
    read = int(body.get("read", 0))

    n = len(words)
    eos_ids = stop_token_ids(TOKENIZER)

    # READ the newly typed words
    reads = []
    while read < n:
        read += 1
        reads.append(words[read - 1])

    # WRITE new target tokens.
    writes = []

    def _record(token_id):
        committed.append(token_id)
        so_far = TOKENIZER.decode(committed, skip_special_tokens=True).strip()
        writes.append({
            "token": TOKENIZER.decode([token_id], skip_special_tokens=True),
            "translation_so_far": so_far,
            "src_read": read,
        })

    if finalize:
        # Flush the rest of the sentence with fast KV-cached generation.
        for token_id in _generate_tail(words[:read], committed, tgt_lang, eos_ids):
            _record(token_id)
    else:
        # Live wait-k: emit up to (read - k + 1) tokens (one per word read past
        # the initial k-word lag), suppressing EOS so it can't end early.
        target = min(max(0, read - k + 1), 256)
        while len(committed) < target:
            next_id = _next_token(words[:read], committed, tgt_lang,
                                  eos_ids, suppress_eos=True)
            if next_id in eos_ids:
                break
            _record(next_id)

    translation = TOKENIZER.decode(committed, skip_special_tokens=True).strip()
    return {
        "committed": committed,
        "read": read,
        "reads": reads,
        "writes": writes,
        "translation_so_far": translation,
        "src_total": n,
        "done": finalize,
    }


@app.post("/api/translate/correct")
async def translate_correct(request: Request):
    """Re-translate the *whole* finished sentence as one offline pass.

    Triggered by the client after a typing pause long enough to mean "the
    sentence is done" (~3 s). Wait-k commits tokens left-to-right before the
    full source is seen, so at low k it tends to mirror the English S-V-O order
    even for SOV Indic targets. This pass sees the complete sentence, so it can
    place the verb last and otherwise fix the order the streaming output guessed
    wrong. It does NOT touch the wait-k decode state, so if the user keeps
    typing the live stream simply resumes from where it was.

    Body: {text: str, target_lang: str} → {translation: str}
    """
    body = await request.json()
    text = (body.get("text", "") or "").strip()
    tgt_lang = LANG_MAP.get(body.get("target_lang", "en"), "English")
    if not text:
        return {"translation": ""}
    return {"translation": await _translate_full(text, tgt_lang)}


_COMET_MODEL = None


def _get_comet():
    """Lazily load COMET once. Runs on CPU so it doesn't fight the 4B model for
    the 6 GB GPU. Returns None if unbabel-comet isn't installed."""
    global _COMET_MODEL
    if _COMET_MODEL is None:
        from comet import download_model, load_from_checkpoint
        path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL = load_from_checkpoint(path)
    return _COMET_MODEL


def _waitk_trace(tgt_tokens: int, src_words: int, k: int):
    """Reconstruct the wait-k READ/WRITE trace for latency metrics: target token
    ``t`` (1-indexed) is written after reading ``min(k + t - 1, S)`` words."""
    trace, read = [], 0
    for t in range(1, tgt_tokens + 1):
        target_read = min(k + t - 1, src_words)
        while read < target_read:
            read += 1
            trace.append(("READ", ""))
        trace.append(("WRITE", ""))
    while read < src_words:
        read += 1
        trace.append(("READ", ""))
    return trace


@app.post("/api/translate/quality")
async def translate_quality(request: Request):
    """Full-sentence (offline) translation + COMET/AL/LAAL for the wait-k output.

    The offline translation is used as the COMET reference, so COMET measures how
    much quality the simultaneous (wait-k) output gives up. AL/LAAL are latency
    metrics derived from the wait-k schedule (tgt_tokens, src_words, k).
    """
    body = await request.json()
    text = body.get("text", "").strip()
    tgt_lang = LANG_MAP.get(body.get("target_lang", "en"), "English")
    waitk_translation = body.get("waitk_translation", "").strip()
    tgt_tokens = int(body.get("tgt_tokens", 0))
    k = max(1, int(body.get("k", 3)))

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    src_words = len(text.split())
    full_translation = _offline_translate(text, tgt_lang)

    # Latency metrics from the wait-k schedule
    trace = _waitk_trace(tgt_tokens, src_words, k)
    al_score = average_lagging(trace, src_words)
    laal_score = laal(trace, src_words)

    # COMET quality: wait-k hypothesis vs offline reference (CPU; may be slow)
    comet = None
    comet_error = None
    if waitk_translation and full_translation:
        try:
            model = _get_comet()
            out = model.predict(
                [{"src": text, "mt": waitk_translation, "ref": full_translation}],
                batch_size=1, gpus=0,
            )
            comet = round(float(out.system_score), 3)
        except ImportError:
            comet_error = "unbabel-comet not installed (pip install unbabel-comet)"
        except Exception as exc:  # noqa: BLE001 — report, don't crash the request
            import traceback
            traceback.print_exc()
            comet_error = f"{type(exc).__name__}: {exc}"

    return {
        "full_translation": full_translation,
        "comet": comet,
        "comet_error": comet_error,
        "al": None if math.isnan(al_score) else round(al_score, 2),
        "laal": None if math.isnan(laal_score) else round(laal_score, 2),
        "src_words": src_words,
        "k": k,
    }


@app.get("/api/examples")
async def examples():
    """Return example sentences for each language."""
    return {
        "examples": {
            "te": [
                "నేను రోజూ ఉదయం పార్కులో నడుస్తాను.",
                "భారతదేశం ప్రపంచంలో అతి పెద్ద ప్రజాస్వామ్య దేశం.",
                "నేను నా స్నేహితులతో సినిమాకు వెళ్ళాను.",
            ],
            "hi": [
                "मैं हर रोज सुबह पार्क में टहलता हूँ.",
                "भारत दुनिया का सबसे बड़ा लोकतंत्र है.",
                "मैंने अपने दोस्तों के साथ फिल्म देखी.",
            ],
            "gu": [
                "હું દરરોજ સવારે પાર્કમાં ચાલું છું.",
                "ભારત વિશ્વનું સૌથી મોટું લોકશાહી છે.",
                "મેં મારા મિત્રો સાથે ફિલ્મ જોઈ.",
            ],
            "ta": [
                "நான் தினமும் காலையில் பூங்காவில் நடப்பேன்.",
                "இந்தியா உலகின் மிகப்பெரிய ஜனநாயக நாடு.",
                "நான் என் நண்பர்களுடன் திரைப்படம் பார்த்தேன்.",
            ],
            "en": [
                "I walk in the park every morning.",
                "India is the largest democracy in the world.",
                "I watched a movie with my friends.",
            ],
        }
    }


# =============================================================================
# Live speech → simultaneous translation (faster-whisper → wait-k / remote MT)
#
# The browser streams 16 kHz mono int16 PCM over a WebSocket. Instead of
# re-transcribing the whole growing utterance every update (which got slower and
# slower as you spoke — O(n²) audio overall — and made the server fall behind
# real time), StreamingASR re-runs Whisper only on the *uncommitted tail* of the
# audio, capped at MAX_WINDOW_S, and freezes words with LocalAgreement-2. So the
# per-update cost stays flat no matter how long the utterance is. The transcript
# is pushed the instant each window is decoded; the stable prefix is translated
# *in the background* with one fast KV-cached generate (so the slow 4B decode
# never stalls the transcript), and on stop the whole sentence is translated in
# one pass — so output appears while you speak yet is never cut off. Speed comes
# from: bounded re-transcription, the GPU (shared by ASR + MT), a *pinned* source
# language (no per-chunk auto-detect), Silero VAD to skip silence, and translating
# only when a new stable word has appeared.
# =============================================================================
SR = 16_000                                   # browser sends 16 kHz mono
RETRANSCRIBE_BYTES = int(SR * 2 * 0.8)        # re-run ASR every ~0.8 s of new audio
MAX_WINDOW_S = 16.0                           # cap audio fed to Whisper per update
ASR_LANGS = {"te", "hi", "gu", "ta", "en"}    # codes faster-whisper accepts here


def _whisper_lang(code):
    """Map a UI source-language code to a pinned Whisper language (or None=auto)."""
    return code if code in ASR_LANGS else None


def _get_whisper():
    """Lazily load faster-whisper once, with a GPU→CPU fallback.

    On the 6 GB card the 4B MT model leaves ~2 GB free — more than enough for a
    memory-light ``int8_float16`` Whisper, which is ~10× faster than CPU. If the
    GPU load fails (OOM / driver), we silently retry on CPU so live speech still
    works. Raises only if the package is missing or both devices fail.
    """
    global WHISPER, WHISPER_DEVICE, WHISPER_COMPUTE
    if WHISPER is None:
        from faster_whisper import WhisperModel
        attempts = [(WHISPER_DEVICE, WHISPER_COMPUTE)]
        if WHISPER_DEVICE == "cuda":
            attempts.append(("cpu", "int8"))          # OOM / driver safety net
        last_exc = None
        for dev, comp in attempts:
            try:
                print(f"[asr] loading faster-whisper '{WHISPER_MODEL}' on "
                      f"{dev} ({comp})...")
                kwargs = {}
                if dev == "cpu":
                    kwargs["cpu_threads"] = min(8, os.cpu_count() or 4)
                WHISPER = WhisperModel(WHISPER_MODEL, device=dev,
                                       compute_type=comp, **kwargs)
                WHISPER_DEVICE, WHISPER_COMPUTE = dev, comp
                break
            except Exception as exc:  # noqa: BLE001 — try the next device
                last_exc = exc
                print(f"[asr] load on {dev} failed "
                      f"({type(exc).__name__}: {exc}); trying next device.")
        if WHISPER is None:
            raise last_exc
    return WHISPER


def _stable_words(text: str, is_final: bool) -> list[str]:
    """Words safe to translate: drop the last (still-unstable) word on partials."""
    words = text.split()
    return words if is_final else (words[:-1] if len(words) > 1 else [])


class StreamingASR:
    """Bounded, incremental ASR for live speech.

    The old path re-transcribed the *entire* growing utterance on every update,
    so per-update cost climbed with utterance length and the server fell behind
    real time. This keeps the cost flat by:

      • running Whisper only on the uncommitted *tail* of the audio, capped at
        MAX_WINDOW_S seconds (older audio is trimmed away once its words are
        committed);
      • committing words with LocalAgreement-2 — a word is frozen only once two
        consecutive transcriptions agree on it (Macháček et al., 2020) — and
        feeding the committed text back as Whisper's ``initial_prompt`` so
        context survives the trim.

    ``transcribe`` returns ``(full_text, language)`` where ``full_text`` is the
    frozen committed prefix plus the current (still-mutable) tail.
    """

    def __init__(self, language=None):
        self.language = language
        self.audio = np.zeros(0, dtype=np.float32)   # uncommitted tail
        self.committed: list[str] = []               # frozen words
        self.prev_tail: list[str] = []               # last hypothesis tail words
        self.lang = language

    def add_audio(self, pcm_bytes: bytes):
        a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, a])

    @property
    def committed_text(self) -> str:
        return " ".join(self.committed)

    def _hyp(self):
        """Transcribe the current (capped) window → list of (word, end_time)."""
        model = _get_whisper()
        max_samples = int(MAX_WINDOW_S * SR)
        if len(self.audio) > max_samples:        # safety valve for endless speech
            self.audio = self.audio[-max_samples:]
        segments, info = model.transcribe(
            self.audio, language=self.language, beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=self.committed_text[-200:] or None,
            word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
        )
        self.lang = info.language
        words = []
        for seg in segments:
            for w in (seg.words or []):
                t = w.word.strip()
                if t:
                    words.append((t, w.end))
        return words

    def transcribe(self, is_final: bool = False):
        hyp = self._hyp()
        if is_final:                              # flush everything on stop
            self.committed.extend(w for w, _ in hyp)
            self.prev_tail = []
            self.audio = np.zeros(0, dtype=np.float32)
            return self.committed_text, self.lang

        # LocalAgreement-2: confirm the longest prefix this run shares with the
        # previous one, but never the final word (still settling).
        n = 0
        for i in range(min(len(hyp), len(self.prev_tail))):
            if hyp[i][0] == self.prev_tail[i]:
                n = i + 1
            else:
                break
        n = min(n, max(0, len(hyp) - 1))

        if n:
            cut_end = hyp[n - 1][1]               # end time of last confirmed word
            self.committed.extend(w for w, _ in hyp[:n])
            cut = min(len(self.audio), int(cut_end * SR))
            self.audio = self.audio[cut:]         # drop the committed audio
            self.prev_tail = [w for w, _ in hyp[n:]]
        else:
            self.prev_tail = [w for w, _ in hyp]

        full = (self.committed_text + " " + " ".join(self.prev_tail)).strip()
        return full, self.lang


async def _transcribe(loop, asr: StreamingASR, is_final: bool):
    """Run the bounded incremental ASR off the event loop so audio keeps draining
    while Whisper works. Returns ``(full_text, language)``."""
    return await loop.run_in_executor(None, asr.transcribe, is_final)


@app.websocket("/api/asr")
async def asr_ws(ws: WebSocket):
    """Live speech → simultaneous translation.

    Two stages share the GPU and run independently so each stays responsive:
      • ASR: bounded incremental Whisper, ~instant on GPU. The transcript is
        pushed the moment each ~0.8 s window is decoded, so words appear in near
        real time regardless of how slow translation is.
      • MT: the stable prefix is translated with a single KV-cached ``generate``
        (the typed path's fast route), in the *background*, so the slow 4B decode
        never blocks the transcript. At most one translation is in flight, so it
        can't pile up. On ``stop`` the whole sentence is translated in one pass —
        the output is always the complete sentence, never cut off mid-way.

    Source language is pinned (no per-chunk auto-detect). Changing it restarts the
    listener in the newly selected language.
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    asr = StreamingASR()            # bounded incremental ASR for this connection
    target_lang = "Telugu"          # default direction is English → Telugu
    k = 3
    recording = False
    pending = 0                     # bytes of new audio since the last transcription
    last_fed = 0                    # stable words already translated
    xl_task: "asyncio.Task | None" = None    # at most one in-flight translation

    async def translate_prefix(feed_words, is_final):
        """Translate the stable prefix and push it to the client."""
        nonlocal last_fed
        try:
            translation = await _translate_full(" ".join(feed_words), target_lang)
            last_fed = len(feed_words)
            await ws.send_json({"type": "final" if is_final else "partial",
                                "translation": translation})
        except Exception:  # noqa: BLE001 — a dropped partial must not kill the socket
            pass

    async def tick(is_final):
        """One ASR update: push the transcript now, translate the prefix after."""
        nonlocal xl_task
        text, lang = await _transcribe(loop, asr, is_final)
        feed = _stable_words(text, is_final)

        if is_final:
            if xl_task and not xl_task.done():
                xl_task.cancel()                     # supersede any live partial
            await ws.send_json({"type": "partial", "transcript": text, "lang": lang})
            translation = await _translate_full(" ".join(feed), target_lang)
            await ws.send_json({"type": "final", "transcript": text, "lang": lang,
                                "translation": translation})
            return

        # Live: transcript first (cheap) so words appear immediately…
        await ws.send_json({"type": "partial", "transcript": text, "lang": lang})
        # …then translate the stable prefix in the background, but only once a new
        # stable word past the wait-k lag has appeared and nothing is mid-flight —
        # so the slow 4B decode never queues up or stalls the live transcript.
        if len(feed) >= k and len(feed) > last_fed and (xl_task is None or xl_task.done()):
            xl_task = asyncio.create_task(translate_prefix(feed, False))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                typ = data.get("type")
                if typ == "start":
                    if xl_task and not xl_task.done():
                        xl_task.cancel()
                    asr = StreamingASR(language=_whisper_lang(data.get("source_lang")))
                    target_lang = LANG_MAP.get(data.get("target_lang", "te"), "Telugu")
                    k = max(1, int(data.get("k", 3)))
                    pending = last_fed = 0
                    xl_task = None
                    recording = True
                    await ws.send_json({"type": "status", "msg": "listening"})
                elif typ == "config":
                    if "target_lang" in data:
                        target_lang = LANG_MAP.get(data["target_lang"], target_lang)
                    if "source_lang" in data:
                        new_lang = _whisper_lang(data["source_lang"])
                        if new_lang != asr.language:   # switch ASR to the new language
                            if xl_task and not xl_task.done():
                                xl_task.cancel()
                            asr = StreamingASR(language=new_lang)
                            pending = last_fed = 0
                    if "k" in data:
                        k = max(1, int(data["k"]))
                elif typ == "stop":
                    recording = False
                    if len(asr.audio) > 0 or asr.committed:
                        await tick(True)
                    else:
                        await ws.send_json({"type": "final", "transcript": "",
                                            "lang": None, "translation": ""})

            elif msg.get("bytes") is not None:
                if not recording:
                    continue
                asr.add_audio(msg["bytes"])
                pending += len(msg["bytes"])
                if pending >= RETRANSCRIBE_BYTES:
                    pending = 0
                    await tick(False)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 — report, don't kill the server
        import traceback
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "msg": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


def _warmup():
    """Run one tiny MT + ASR pass at startup so the *first* live utterance isn't
    slowed by one-off CUDA kernel autotuning (the cold MT decode costs ~1 s extra).
    Best-effort; only warms what's already resident on the GPU."""
    if MODEL is not None:                  # local mode: warm the decode path
        try:
            _offline_translate("hello", "Telugu")
        except Exception:  # noqa: BLE001
            pass
    if WHISPER is not None:                 # warm the ASR forward pass
        try:
            WHISPER.transcribe(np.zeros(SR, dtype=np.float32), language="en")
        except Exception:  # noqa: BLE001
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="sarvamai/sarvam-translate")
    p.add_argument("--adapter", default=None)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--quantize-4bit", action="store_true", default=False,
                   help="Force 4-bit quantization (for GPUs with < 8 GB VRAM)")
    p.add_argument("--no-quantize", action="store_true", default=False,
                   help="Force full precision (needs 8 GB+ VRAM)")
    p.add_argument("--whisper-model", default="base",
                   help="faster-whisper model for live speech ASR "
                        "(tiny/base/small/medium; multilingual). 'base' is the "
                        "snappy default; 'small' is more accurate for Indian "
                        "languages but slower (fine on GPU).")
    p.add_argument("--whisper-compute", default="int8",
                   help="faster-whisper compute type (int8 / float16 / float32). "
                        "Ignored when --whisper-device=auto picks the GPU.")
    p.add_argument("--whisper-device", default="auto",
                   help="auto | cpu | cuda. 'auto' uses the GPU when the remote "
                        "LLM has freed it, otherwise CPU.")
    p.add_argument("--no-asr", action="store_true", default=False,
                   help="Skip loading the ASR model (typed demo only)")
    # ---- Remote translation backend (primary) ----
    p.add_argument("--backend", choices=["auto", "remote", "local"],
                   default=os.environ.get("TRANSLATE_BACKEND", "auto"),
                   help="auto: use the remote LLM if reachable else local; "
                        "remote: force remote (local only as runtime fallback); "
                        "local: the original on-device 4B model.")
    p.add_argument("--api-url",
                   default=os.environ.get("TRANSLATE_API_URL",
                                          "http://64.247.196.173:8080/v1"),
                   help="Base URL of the OpenAI-compatible translation LLM.")
    p.add_argument("--api-key", default=os.environ.get("TRANSLATE_API_KEY", "EMPTY"))
    p.add_argument("--api-model", default=os.environ.get("TRANSLATE_API_MODEL"),
                   help="Model id to request (default: auto-discover via /v1/models).")
    return p.parse_args()


def main():
    global MODEL, TOKENIZER, DEVICE, WHISPER_MODEL, WHISPER_COMPUTE, WHISPER_DEVICE
    global TRANSLATE_BACKEND, API_URL, API_KEY, API_MODEL, USE_REMOTE, REMOTE_OK
    global _LOCAL_ARGS
    args = parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRANSLATE_BACKEND = args.backend
    API_URL = args.api_url.rstrip("/")
    API_KEY = args.api_key
    API_MODEL = args.api_model

    # Auto-detect quantization need based on available VRAM
    if args.no_quantize:
        use_4bit = False
    elif args.quantize_4bit:
        use_4bit = True
    elif DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        use_4bit = vram_gb < 8.0
        print(f"[server] Detected {vram_gb:.1f} GB VRAM → "
              f"{'4-bit quantization' if use_4bit else 'full precision'}")
    else:
        use_4bit = False
    # Remember how to load the local model so it can be a lazy fallback later.
    _LOCAL_ARGS = {"base": args.base, "adapter": args.adapter, "use_4bit": use_4bit}

    # ---- Choose the translation engine ----
    if TRANSLATE_BACKEND in ("auto", "remote"):
        print(f"[server] Probing remote LLM at {API_URL} ...")
        REMOTE_OK = _probe_remote()
        if REMOTE_OK:
            USE_REMOTE = True
            print(f"[server] Using REMOTE LLM (model={API_MODEL or 'default'}). "
                  f"Local 4B NOT loaded → GPU free for fast ASR.")
        elif TRANSLATE_BACKEND == "remote":
            USE_REMOTE = True   # forced; load local lazily only if a request fails
            print("[server] Remote forced but probe failed; will retry per request "
                  "and fall back to the local model on demand.")
        else:
            print("[server] Remote unreachable → falling back to the LOCAL model.")

    if not USE_REMOTE:
        print(f"[server] Loading local model on {DEVICE} (4-bit={use_4bit})...")
        # eager is the safe, proven config for this model.
        MODEL, TOKENIZER = load_model(
            args.base, args.adapter, device=DEVICE, quantize_4bit=use_4bit,
            attn_implementation="eager",
        )
        print("[server] Local model loaded.")

    # ---- faster-whisper device ----
    # Prefer the GPU whenever CUDA exists. In remote mode the GPU is idle, so we
    # use float16 (fastest). In local mode the 4B model is resident, so we use the
    # memory-light int8_float16 — it still fits in the ~2 GB the 4B model leaves
    # free and is ~10× faster than CPU (which was the old, "extremely slow" path).
    # _get_whisper() falls back to CPU automatically if the GPU load OOMs.
    WHISPER_MODEL = args.whisper_model
    if args.whisper_device == "auto":
        if torch.cuda.is_available():
            WHISPER_DEVICE = "cuda"
            WHISPER_COMPUTE = "float16" if USE_REMOTE else "int8_float16"
        else:
            WHISPER_DEVICE, WHISPER_COMPUTE = "cpu", "int8"
    else:
        WHISPER_DEVICE, WHISPER_COMPUTE = args.whisper_device, args.whisper_compute

    if not args.no_asr:
        try:
            _get_whisper()                # preload so the first utterance is instant
            print(f"[server] ASR ready (faster-whisper '{WHISPER_MODEL}' on "
                  f"{WHISPER_DEVICE}/{WHISPER_COMPUTE}). Live speech enabled.")
        except Exception as exc:  # noqa: BLE001 — typed demo still works without ASR
            print(f"[server] WARNING: ASR unavailable ({type(exc).__name__}: {exc}).")
            print("[server] Install it with: pip install faster-whisper")

    _warmup()                              # first utterance is then full-speed
    print(f"[server] Open http://localhost:{args.port} in your browser")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

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
import math
import sys
from pathlib import Path

from fastapi import FastAPI, Request
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "index.html").read_text()


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
    committed = [int(x) for x in body.get("committed", [])]
    read = int(body.get("read", 0))
    finalize = bool(body.get("finalize", False))

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
    return p.parse_args()


def main():
    global MODEL, TOKENIZER, DEVICE
    args = parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    print(f"[server] Loading model on {DEVICE} (4-bit={use_4bit})...")
    # eager is the safe, proven config for this model; the main speedup is the
    # incremental /step decoding plus loading in bf16 (see load.py).
    MODEL, TOKENIZER = load_model(
        args.base, args.adapter, device=DEVICE, quantize_4bit=use_4bit,
        attn_implementation="eager",
    )
    print(f"[server] Model loaded. Starting server on {args.host}:{args.port}")
    print(f"[server] Open http://localhost:{args.port} in your browser")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

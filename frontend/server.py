"""FastAPI backend for the Simultaneous MT presentation frontend.

Serves translation requests (full-sentence and streaming wait-k) and
pre-computed evaluation metrics. Runs on port 8080.

Usage:
    python server.py
    python server.py --adapter checkpoints_multilang/final --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

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
    wait_k_decode,
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


@app.post("/api/translate")
async def translate(request: Request):
    """Full-sentence (offline) translation."""
    body = await request.json()
    text = body.get("text", "").strip()
    tgt_lang = LANG_MAP.get(body.get("target_lang", "en"), "English")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    model_device = next(MODEL.parameters()).device
    prompt = build_prompt(TOKENIZER, text, tgt_lang)
    enc = TOKENIZER(prompt, return_tensors="pt").to(model_device)

    with torch.inference_mode():
        gen = MODEL.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            pad_token_id=TOKENIZER.pad_token_id,
        )
    out = TOKENIZER.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    translation = out.split("<end_of_turn>")[0].strip()

    return {"translation": translation, "policy": "full", "target_language": tgt_lang}


@app.get("/api/translate/stream")
async def translate_stream(text: str, target_lang: str = "en", k: int = 3):
    """SSE-streamed wait-k translation with READ/WRITE events."""
    text = text.strip()
    tgt_lang = LANG_MAP.get(target_lang, "English")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    async def event_generator():
        eos_ids = stop_token_ids(TOKENIZER)
        src_words = text.split()
        S = len(src_words)
        committed = []
        prev_read = 0
        model_device = next(MODEL.parameters()).device

        for t in range(1, 257):
            num_src = min(k + t - 1, S)

            # Emit READ events
            while prev_read < num_src:
                prev_read += 1
                word = src_words[prev_read - 1]
                yield {
                    "event": "read",
                    "data": json.dumps({
                        "word": word,
                        "src_read": prev_read,
                        "src_total": S,
                    }),
                }
                await asyncio.sleep(0.15)  # Delay for animation

            # Generate next token
            prompt = build_prompt(
                TOKENIZER, " ".join(src_words[:num_src]), tgt_lang
            )
            prompt_ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(model_device)
            if committed:
                tail = torch.tensor(
                    [committed], device=model_device, dtype=prompt_ids.dtype
                )
                input_ids = torch.cat([prompt_ids, tail], dim=1)
            else:
                input_ids = prompt_ids

            with torch.inference_mode():
                logits = MODEL(input_ids=input_ids).logits[0, -1]
            if num_src < S:
                for eid in eos_ids:
                    logits[eid] = float("-inf")

            next_id = int(logits.argmax())
            if next_id in eos_ids:
                yield {
                    "event": "stop",
                    "data": json.dumps({"reason": "eos"}),
                }
                break

            committed.append(next_id)
            token_text = TOKENIZER.decode([next_id], skip_special_tokens=True)
            full_so_far = TOKENIZER.decode(committed, skip_special_tokens=True).strip()

            yield {
                "event": "write",
                "data": json.dumps({
                    "token": token_text,
                    "translation_so_far": full_so_far,
                    "tgt_tokens": len(committed),
                }),
            }
            await asyncio.sleep(0.1)

        # Final summary
        translation = TOKENIZER.decode(committed, skip_special_tokens=True).strip()
        yield {
            "event": "done",
            "data": json.dumps({
                "translation": translation,
                "k": k,
                "src_words": S,
                "tgt_tokens": len(committed),
            }),
        }

    return EventSourceResponse(event_generator())


@app.post("/api/translate/compare")
async def compare(request: Request):
    """Side-by-side comparison: full-sentence vs wait-k with latency metrics."""
    import math
    body = await request.json()
    text = body.get("text", "").strip()
    tgt_lang_code = body.get("target_lang", "en")
    tgt_lang = LANG_MAP.get(tgt_lang_code, "English")
    k = body.get("k", 3)

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    # 1. Full-sentence (offline) translation
    model_device = next(MODEL.parameters()).device
    prompt = build_prompt(TOKENIZER, text, tgt_lang)
    enc = TOKENIZER(prompt, return_tensors="pt").to(model_device)
    with torch.inference_mode():
        gen = MODEL.generate(
            **enc, max_new_tokens=256, do_sample=False,
            num_beams=1, pad_token_id=TOKENIZER.pad_token_id,
        )
    out = TOKENIZER.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    full_translation = out.split("<end_of_turn>")[0].strip()

    # 2. Wait-K (simultaneous) translation
    waitk_translation, trace = wait_k_decode(
        MODEL, TOKENIZER, text, k=k, target_language=tgt_lang, max_target_tokens=256
    )

    # 3. Compute latency metrics
    num_src_words = len(text.split())
    al_score = average_lagging(trace, num_src_words)
    laal_score = laal(trace, num_src_words)
    al_val = None if math.isnan(al_score) else round(al_score, 2)
    laal_val = None if math.isnan(laal_score) else round(laal_score, 2)

    return {
        "full_translation": full_translation,
        "waitk_translation": waitk_translation,
        "metrics": {"al": al_val, "laal": laal_val, "src_words": num_src_words, "k": k},
    }


@app.get("/api/metrics")
async def metrics():
    """Return pre-computed evaluation metrics from metrics_summary.json."""
    metrics_paths = [
        ROOT / "eval_results" / "metrics_summary.json",
        ROOT / "eval_results_adaptive" / "metrics_summary.json",
    ]
    for p in metrics_paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)

    # Return sample data if no evaluation has been run yet
    return {
        "base": "sarvamai/sarvam-translate",
        "adapter": None,
        "note": "No evaluation results found. Run run_full_pipeline.sh first.",
        "results": [],
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
    MODEL, TOKENIZER = load_model(
        args.base, args.adapter, device=DEVICE, quantize_4bit=use_4bit
    )
    print(f"[server] Model loaded. Starting server on {args.host}:{args.port}")
    print(f"[server] Open http://localhost:{args.port} in your browser")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

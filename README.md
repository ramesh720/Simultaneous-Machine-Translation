# Simultaneous Machine Translation

Fine-tune [`sarvamai/sarvam-translate`](https://huggingface.co/sarvamai/sarvam-translate) (Gemma3-4B) for **simultaneous machine translation** of Indian languages using the **Wait-K** policy with **SimulMask** attention masking. Includes multi-GPU training, evaluation across fixed and adaptive policies, and an interactive browser demo.

**Languages:** Telugu, Hindi, Gujarati, Tamil ↔ English (4 languages, 8 directions)  
**Hardware target:** single RTX 3060 6 GB (auto 4-bit quantization) up to multi-GPU via DDP + Accelerate

---

## How it works

The base model was trained on full sentences. Decoding under a streaming wait-k policy is a train/test mismatch — quality drops, especially for SOV→SVO reordering (Telugu→English). Two complementary training strategies fix this:

- **Wait-K prefix training** — with probability `full_sentence_prob` train on the full pair; otherwise sample a source prefix of `j` words and supervise the wait-k aligned target prefix of `i = j − k + 1` words (no EOS). Loss is computed only on target tokens.
- **SimulMask** (EMNLP 2024) — instead of truncating the source, constrain the transformer attention mask so that target token `t` can only attend to the first `min(k + t − 1, S)` source tokens. Full source/target pairs are used; the model retains full-sentence capability.

---

## Repository layout

```
.
├── waitk_finetune/             # training + evaluation library
│   ├── configs/
│   │   ├── config.yaml         # Telugu↔English single-language config
│   │   └── config_multilang.yaml
│   ├── src/
│   │   ├── waitk.py            # prompt format, wait-k decode, AL / LAAL metrics
│   │   ├── simulmask.py        # SimulMask attention-mask builder
│   │   ├── data.py             # WaitKDataModule (on-the-fly prefix sampling)
│   │   ├── module.py           # PyTorch Lightning module + LoRA
│   │   ├── load.py             # model loader (bf16, 4-bit, adapter)
│   │   ├── comet_eval.py       # COMET scoring helper
│   │   └── adaptive.py         # Local Agreement + Confidence threshold decoders
│   ├── compare.py              # side-by-side comparison of two eval outputs
│   ├── evaluate.py             # BLEU / chrF++ / AL on IN22 (single script)
│   ├── inference.py            # translate a single sentence
│   ├── train.py                # training entry point
│   └── requirements.txt
│
├── eval_simulmt.py             # multi-GPU eval: full + wait-k policies
├── eval_adaptive.py            # multi-GPU eval: adds Local Agreement + Confidence
├── prepare_in22_data.py        # download and format IN22-Conv/Gen test sets
├── prepare_multilang_data.py   # build multi-language training TSV from BPCC
├── plot_results.py             # quality–latency plots from metrics_summary.json
├── streaming_st.py             # live mic → faster-whisper → wait-k MT pipeline
├── run_eval.sh                 # one-command evaluation run
├── run_full_pipeline.sh        # end-to-end: data → train → eval → frontend
│
├── eval_data/                  # IN22-Conv and IN22-Gen test sets (JSON)
├── eval_results/               # evaluation outputs + metrics_summary.json
│
└── frontend/
    ├── server.py               # FastAPI backend (port 8080)
    ├── index.html              # interactive demo UI
    ├── app.js
    └── style.css
```

---

## Setup

```bash
pip install -r waitk_finetune/requirements.txt
```

Requires `transformers >= 4.50` for Gemma3. For GPUs with < 8 GB VRAM, 4-bit quantization is auto-detected at load time (no flag needed).

---

## Data

### Training data

Uses BPCC TSV files (`tel_Telu.tsv`, etc.) with columns `src_lang, tgt_lang, src, tgt`.  
For multi-language training, build a combined TSV first:

```bash
python prepare_multilang_data.py --out_path multilang_train.tsv \
    --langs te,hi,gu,ta --max_per_direction 50000
```

### Evaluation data (IN22)

```bash
python prepare_in22_data.py --out_dir eval_data --langs te,hi,gu,ta
```

This downloads IN22-Conv and IN22-Gen for each language pair from HuggingFace (requires access).

---

## Training

Single GPU (Telugu→English, wait-3):
```bash
cd waitk_finetune
python train.py --config configs/config.yaml trainer.devices=1
```

Multi-language, multi-GPU (DDP):
```bash
python train.py --config configs/config_multilang.yaml trainer.devices=4
```

Any field is overridable on the CLI:
```bash
python train.py --config configs/config.yaml \
    waitk.k=5 training.lr=5e-5 lora.r=32 wandb.enabled=false
```

The final LoRA adapter is saved to `checkpoints/final/` (or `checkpoints_multilang/final/`).

**Key config defaults:** LoRA r=16 α=32, wait-k=3, full_sentence_prob=0.3, batch_size=4, 3 epochs, bf16.

---

## Evaluation

### Fixed policies (full-sentence and wait-k)

Single GPU:
```bash
python eval_simulmt.py \
    --data_dir eval_data --out_dir eval_results \
    --langs te --datasets conv gen \
    --directions x2e --policies full waitk --k 3 5 7 --limit 200
```

Multi-GPU (4 GPUs, all languages/directions):
```bash
bash run_eval.sh
# or directly:
accelerate launch --num_processes 4 eval_simulmt.py \
    --data_dir eval_data --out_dir eval_results \
    --langs te hi gu ta --datasets conv gen --directions x2e e2x \
    --policies full waitk --k 1 3 5 7
```

### Adaptive policies

Extends the above with two training-free adaptive decoders:

- **Local Agreement (LA-n)** — commit a token once the last `n` re-translations agree.
- **Confidence threshold** — WRITE the next greedy token only when its probability ≥ τ, else READ more source.

```bash
accelerate launch --num_processes 4 eval_adaptive.py \
    --langs te hi gu ta --datasets conv gen --directions x2e e2x \
    --policies full waitk la conf \
    --k 1 3 5 7 --n 2 3 --tau 0.4 0.6 0.8
```

Metrics reported: **BLEU** (sacrebleu `intl` tokenizer, matching IN22 benchmark), **COMET** (Unbabel/wmt22-comet-da), **Average Lagging (AL)**, **Length-Adaptive AL (LAAL)**.

### Baseline results (base model, no fine-tuning, Telugu→English, n=10)

| Dataset | Policy  | BLEU  | AL   | LAAL |
|---------|---------|-------|------|------|
| conv    | full    | 29.76 | —    | —    |
| conv    | wait-3  | 30.02 | 3.10 | 0.80 |
| conv    | wait-5  | 28.47 | 3.85 | 0.55 |
| conv    | wait-7  | 29.76 | 4.00 | 0.49 |
| gen     | full    | 18.58 | —    | —    |
| gen     | wait-3  | 18.26 | 5.71 | 2.75 |
| gen     | wait-5  | 18.90 | 6.84 | 3.07 |
| gen     | wait-7  | 18.68 | 8.51 | 3.03 |

---

## Single-sentence inference

```bash
cd waitk_finetune

# wait-k with READ/WRITE trace
python inference.py --text "నేను రోజూ ఉదయం పార్కులో నడుస్తాను." \
    --policy waitk --k 3 --adapter checkpoints/final --trace

# full-sentence offline translation
python inference.py --text "..." --policy full
```

---

## Interactive frontend demo

A FastAPI backend serves a browser UI at `http://localhost:8080`. Type a sentence and the wait-k translation appears token by token. After you stop typing, a "Check full sentence" button shows COMET / AL / LAAL comparing the simultaneous output against offline translation.

```bash
# base model (auto 4-bit on 6 GB GPU)
python frontend/server.py

# with a fine-tuned adapter
python frontend/server.py --adapter waitk_finetune/checkpoints_multilang/final

# force 4-bit or full precision
python frontend/server.py --quantize-4bit
python frontend/server.py --no-quantize
```

API endpoints:
- `GET /` — demo UI
- `GET /api/languages` — supported languages and directions
- `GET /api/examples` — example sentences per language
- `POST /api/translate/step` — stateless incremental wait-k step (one call per word typed)
- `POST /api/translate/quality` — full-sentence translation + COMET / AL / LAAL

---

## Live speech-to-text translation

Streams microphone audio through VAD → faster-whisper ASR → wait-k MT:

```bash
pip install faster-whisper sounddevice webrtcvad numpy
python streaming_st.py --k 3 --tgt-lang Telugu
python streaming_st.py --k 5 --whisper-model small.en --adapter waitk_finetune/checkpoints/final
```

---

## Full pipeline (data → train → eval → demo)

```bash
bash run_full_pipeline.sh              # full end-to-end
bash run_full_pipeline.sh --skip-train # skip training, use existing adapter
bash run_full_pipeline.sh --frontend-only  # just start the demo server
```

---

## Notes

- **4-bit quantization** is auto-applied when < 8 GB VRAM is detected, keeping the full Gemma3-4B model within a 6 GB budget.
- **SimulMask vs prefix truncation:** SimulMask uses the full source/target pair with a constrained attention mask — better KV-cache efficiency and less train/test mismatch than cutting the source.
- **AL vs LAAL:** Average Lagging (AL) penalizes waiting more for longer sentences; Length-Adaptive AL (LAAL) normalizes for sentence length, giving a fairer comparison across sentence lengths.
- **DDP + LoRA:** `strategy: ddp_find_unused_parameters_true` is required because the frozen base model produces unused-parameter gradients.
- **Decoding cost:** `wait_k_decode` re-runs a forward pass per token (no KV-cache reuse) — acceptable for evaluation subsets. The frontend uses incremental `_next_token` calls to amortize this.

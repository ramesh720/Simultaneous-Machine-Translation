# Wait-k Fine-tuning for sarvam-translate (Telugu → English)

Fine-tune [`sarvamai/sarvam-translate`](https://huggingface.co/sarvamai/sarvam-translate)
(a Gemma3-based translation model) for **simultaneous machine translation** using the
**wait-k** prefix-to-prefix training scheme, with PyTorch Lightning, LoRA, multi-GPU
(DDP) support, Weights & Biases logging, and YAML configuration.

> Adaptive READ/WRITE policies are intentionally out of scope here — this trains a
> *fixed* wait-k model first. Adaptive policies can reuse the same data/model code later.

## Why wait-k *training*?

The base model only ever saw full sentences. Decoding it under a streaming wait-k
policy (commit a token after seeing only `k + t − 1` source words) is a train/test
mismatch — quality drops, especially for Telugu→English where the verb is
sentence-final (SOV → SVO reordering). Wait-k training fixes this by teaching the
model to translate **partial source → partial target**:

- With probability `full_sentence_prob`, train on the full `(source → target)` pair
  (keeps full-sentence ability + EOS).
- Otherwise sample a source prefix of `j` words and supervise the wait-k aligned
  target prefix of `i = j − k + 1` words (no EOS — it's a partial output).

Loss is computed only on target tokens (the prompt is masked).

## Layout

```
waitk_finetune/
├── configs/config.yaml     # all hyperparameters; override from CLI
├── src/
│   ├── waitk.py            # prompt format, prefix builder, wait-k decode, Average Lagging
│   ├── data.py             # WaitKDataModule + dataset (on-the-fly prefix sampling)
│   ├── module.py           # LightningModule (+ LoRA)
│   └── load.py             # load base (+ adapter) for eval/inference
├── train.py                # training entry point
├── evaluate.py             # BLEU / chrF++ / Average Lagging on IN22
├── inference.py            # translate a single sentence (full or wait-k)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd waitk_finetune
pip install -r requirements.txt
```

A recent `transformers` (≥ 4.50) is required for the `gemma3` architecture.

## Data

Uses the BPCC `tel_Telu.tsv` (columns: `src_lang, tgt_lang, src, tgt`, where `src`
is English and `tgt` is Telugu). For **Telugu→English** the config reads Telugu
(`tgt`) as the source and English (`src`) as the target:

```yaml
data:
  source_column: tgt        # Telugu -> model input
  target_column: src        # English -> model output
  target_language: English
```

To train **English→Telugu** instead, swap the columns and set
`target_language: Telugu`.

## Train

Single GPU:
```bash
python train.py --config configs/config.yaml trainer.devices=1
```

Multi-GPU (DDP, e.g. 4 GPUs, wait-5):
```bash
python train.py --config configs/config.yaml trainer.devices=4 waitk.k=5
```

Any field is overridable on the CLI (OmegaConf dotlist):
```bash
python train.py --config configs/config.yaml \
    training.lr=5e-5 training.batch_size=4 lora.r=32 wandb.enabled=false
```

The final LoRA adapter is written to `checkpoints/final/`, and the best-by-`val/loss`
checkpoints to `checkpoints/`.

## Evaluate

Compare full-sentence vs wait-k on the IN22 sets (BLEU uses the `intl` tokenizer to
match the IN22 benchmark; chrF++ is also reported):

```bash
# Baseline (no fine-tuning), full sentence
python evaluate.py --dataset ../in22_conv_te_en.json --policy full

# Fine-tuned, wait-k sweep on a 200-sentence subset
python evaluate.py --dataset ../in22_conv_te_en.json --policy waitk \
    --k 1 3 5 7 --adapter checkpoints/final --limit 200 --out conv_waitk.json
```

`Average Lagging (AL)` is reported per `k` so you can read off the quality–latency
trade-off (lower AL = more simultaneous).

## Inference

```bash
# wait-k with a step-by-step READ/WRITE trace
python inference.py --text "నేను రోజూ ఉదయం పార్కులో నడుస్తాను." \
    --policy waitk --k 3 --adapter checkpoints/final --trace

# offline full-sentence translation
python inference.py --text "..." --policy full --adapter checkpoints/final
```

## Notes & caveats

- **Word-based wait-k alignment.** Prefix boundaries are counted in *words* on both
  sides (the source/target token ratio differs), which is an approximation but matches
  how `wait_k_decode` reads the source. Good enough as a training signal.
- **Decoding cost.** `wait_k_decode` re-runs a forward pass per token (no KV-cache
  reuse) for clarity — fine for evaluation subsets, slow for the full set. Raise
  `--limit` as your time budget allows.
- **bf16 + LoRA.** Defaults load the base in bf16 and train LoRA with `bf16-true`.
  For larger batch sizes, enable QLoRA (4-bit) — note 4-bit + DDP needs extra care; for
  multi-GPU prefer plain bf16 LoRA.
- **DDP + LoRA.** `strategy: ddp_find_unused_parameters_true` because the frozen base
  produces unused-parameter grads.
```

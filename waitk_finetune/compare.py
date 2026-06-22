"""Compare BLEU / chrF++ / Average Lagging *before vs after* fine-tuning,
under full-sentence and wait-k decoding for several k.

Loads one model at a time (base, then fine-tuned) and frees it in between so a
single ~16 GB GPU is enough. The fine-tuned path accepts either a Lightning
``.ckpt`` or a PEFT adapter directory.

Examples:
    # Compare base vs a Lightning checkpoint on a 200-sentence subset
    python compare.py --dataset ../in22_conv_te_en.json \
        --finetuned checkpoints/waitk-1-757-0.802.ckpt \
        --k 1 3 5 7 --limit 200 --out compare_conv.json --plot

    # Use a merged adapter dir, evaluate the full set
    python compare.py --dataset ../in22_gen_te_en.json \
        --finetuned checkpoints/final --k 1 3 5
"""
import argparse
import gc
import json

import sacrebleu
import torch
from tqdm.auto import tqdm

from src.load import load_from_ckpt, load_model
from src.waitk import average_lagging, build_prompt, load_pairs, wait_k_decode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="sarvamai/sarvam-translate")
    p.add_argument("--finetuned", required=True,
                   help="Lightning .ckpt or PEFT adapter dir")
    p.add_argument("--dataset", required=True, help="IN22-style json (input/reference)")
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--tgt-lang", default="English")
    p.add_argument("--limit", type=int, default=None, help="evaluate first N sentences")
    p.add_argument("--batch-size", type=int, default=4, help="full-sentence batch size")
    p.add_argument("--tokenize", default="intl", help="sacrebleu tokenizer (intl matches IN22)")
    p.add_argument("--out", default="compare_results.json")
    p.add_argument("--plot", action="store_true", help="also save a BLEU-vs-AL plot")
    return p.parse_args()


@torch.inference_mode()
def translate_full(model, tokenizer, sources, tgt_lang, batch_size):
    outputs = []
    for start in tqdm(range(0, len(sources), batch_size), desc="full", leave=False):
        batch = sources[start:start + batch_size]
        texts = [build_prompt(tokenizer, x, tgt_lang) for x in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False, num_beams=1,
                             pad_token_id=tokenizer.pad_token_id)
        gen = gen[:, enc.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(o.split("<end_of_turn>")[0].strip() for o in decoded)
    return outputs


def score(hyps, references, tokenize):
    bleu = sacrebleu.corpus_bleu(hyps, [references], tokenize=tokenize).score
    chrf = sacrebleu.corpus_chrf(hyps, [references], word_order=2).score
    return bleu, chrf


def run_all_policies(model, tokenizer, sources, references, args, tag):
    """Full-sentence + every wait-k policy for one loaded model."""
    rows = []

    hyps = translate_full(model, tokenizer, sources, args.tgt_lang, args.batch_size)
    bleu, chrf = score(hyps, references, args.tokenize)
    rows.append({"model": tag, "policy": "full", "k": None,
                 "bleu": bleu, "chrf": chrf, "AL": None})
    print(f"[{tag}] full     BLEU={bleu:6.2f}  chrF++={chrf:6.2f}")

    for k in args.k:
        hyps, als = [], []
        for src in tqdm(sources, desc=f"[{tag}] wait-{k}", leave=False):
            hyp, trace = wait_k_decode(model, tokenizer, src, k=k,
                                       target_language=args.tgt_lang, verbose=False)
            hyps.append(hyp)
            als.append(average_lagging(trace, len(src.split())))
        bleu, chrf = score(hyps, references, args.tokenize)
        mean_al = sum(als) / len(als)
        rows.append({"model": tag, "policy": f"wait-{k}", "k": k,
                     "bleu": bleu, "chrf": chrf, "AL": mean_al})
        print(f"[{tag}] wait-{k:<2} BLEU={bleu:6.2f}  chrF++={chrf:6.2f}  AL={mean_al:6.2f}")
    return rows


def load_finetuned(path, base):
    return load_from_ckpt(path) if path.endswith(".ckpt") else load_model(base, adapter_path=path)


def print_table(rows, k_values):
    """Side-by-side BLEU/AL: base vs fine-tuned, with delta."""
    by = {(r["model"], r["policy"]): r for r in rows}
    order = ["full"] + [f"wait-{k}" for k in k_values]
    print("\n" + "=" * 64)
    print(f"{'policy':<10}{'BLEU base':>11}{'BLEU ft':>10}{'ΔBLEU':>8}{'AL base':>9}{'AL ft':>8}")
    print("-" * 64)
    for pol in order:
        b = by.get(("base", pol))
        f = by.get(("finetuned", pol))
        if not b or not f:
            continue
        d = f["bleu"] - b["bleu"]
        alb = "  -  " if b["AL"] is None else f"{b['AL']:.2f}"
        alf = "  -  " if f["AL"] is None else f"{f['AL']:.2f}"
        print(f"{pol:<10}{b['bleu']:>11.2f}{f['bleu']:>10.2f}{d:>+8.2f}{alb:>9}{alf:>8}")
    print("=" * 64)


def maybe_plot(rows, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    plt.figure(figsize=(7, 5))
    for tag, color in [("base", "tab:gray"), ("finetuned", "tab:blue")]:
        pts = sorted((r for r in rows if r["model"] == tag and r["k"] is not None),
                     key=lambda r: r["AL"])
        plt.plot([r["AL"] for r in pts], [r["bleu"] for r in pts], "o-",
                 color=color, label=f"{tag} (wait-k)")
        for r in pts:
            plt.annotate(f"k={r['k']}", (r["AL"], r["bleu"]),
                         textcoords="offset points", xytext=(5, 4), fontsize=8, color=color)
        full = next(r for r in rows if r["model"] == tag and r["policy"] == "full")
        plt.axhline(full["bleu"], ls="--", color=color, alpha=0.6,
                    label=f"{tag} full = {full['bleu']:.1f}")
    plt.xlabel("Average Lagging  (lower = more simultaneous)")
    plt.ylabel(f"BLEU ({args.tokenize})")
    plt.title(f"Wait-k quality-latency: base vs fine-tuned\n({args.dataset})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png = args.out.rsplit(".", 1)[0] + ".png"
    plt.savefig(png, dpi=120)
    print(f"saved {png}")


def free_gpu():
    """Release any freed tensors back to the CUDA allocator.

    Note: callers must drop their *own* references (del) first — passing a
    model into a helper and deleting it there only clears the local name.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    args = parse_args()
    sources, references = load_pairs(args.dataset)
    if args.limit:
        sources, references = sources[:args.limit], references[:args.limit]
    print(f"Evaluating on {len(sources)} sentences from {args.dataset}\n")

    # --- Base (one model on GPU at a time) ---
    model, tokenizer = load_model(args.base, adapter_path=None)
    base_rows = run_all_policies(model, tokenizer, sources, references, args, "base")
    # Drop the references *here* so the base model actually leaves the GPU
    # before the fine-tuned model is loaded (otherwise both coexist -> OOM).
    del model, tokenizer
    free_gpu()
    if torch.cuda.is_available():
        print(f"[mem] after freeing base: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    # --- Fine-tuned ---
    model, tokenizer = load_finetuned(args.finetuned, args.base)
    ft_rows = run_all_policies(model, tokenizer, sources, references, args, "finetuned")
    del model, tokenizer
    free_gpu()

    rows = base_rows + ft_rows
    print_table(rows, args.k)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "base": args.base, "finetuned": args.finetuned,
                   "n": len(sources), "tokenize": args.tokenize, "rows": rows}, f, indent=2)
    print(f"\nsaved {args.out}")

    if args.plot:
        maybe_plot(rows, args)


if __name__ == "__main__":
    main()

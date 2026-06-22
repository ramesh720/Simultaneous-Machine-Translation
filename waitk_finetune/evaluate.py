"""Evaluate a model on IN22-style json (Telugu -> English) under full-sentence
or wait-k decoding, reporting BLEU + chrF (+ Average Lagging for wait-k).

Examples:
    # Full-sentence baseline
    python evaluate.py --dataset ../in22_conv_te_en.json --policy full

    # Fine-tuned adapter under wait-3
    python evaluate.py --dataset ../in22_conv_te_en.json --policy waitk --k 3 \
        --adapter checkpoints/final

    # Sweep several k values
    python evaluate.py --dataset ../in22_conv_te_en.json --policy waitk \
        --k 1 3 5 7 --adapter checkpoints/final --limit 200 --out results.json
"""
import argparse
import json

import sacrebleu
import torch
from tqdm.auto import tqdm

from src.load import load_model
from src.waitk import average_lagging, build_prompt, load_pairs, wait_k_decode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="sarvamai/sarvam-translate")
    p.add_argument("--adapter", default=None, help="path to LoRA adapter / fine-tuned dir")
    p.add_argument("--dataset", required=True, help="IN22-style json with input/reference")
    p.add_argument("--policy", choices=["full", "waitk"], default="full")
    p.add_argument("--k", type=int, nargs="+", default=[3], help="wait-k value(s)")
    p.add_argument("--tgt-lang", default="English")
    p.add_argument("--limit", type=int, default=None, help="evaluate first N sentences")
    p.add_argument("--batch-size", type=int, default=8, help="full-sentence batch size")
    p.add_argument("--tokenize", default="intl", help="sacrebleu tokenizer (intl matches IN22)")
    p.add_argument("--out", default='results_finetuned', help="write results json here")
    return p.parse_args()


@torch.inference_mode()
def translate_full(model, tokenizer, sources, tgt_lang, batch_size):
    outputs = []
    for start in tqdm(range(0, len(sources), batch_size), desc="full"):
        batch = sources[start:start + batch_size]
        texts = [build_prompt(tokenizer, x, tgt_lang) for x in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False, num_beams=1,
                             pad_token_id=tokenizer.pad_token_id)
        gen = gen[:, enc.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        # gemma3 may emit the end-of-turn marker as text; strip it.
        outputs.extend(o.split("<end_of_turn>")[0].strip() for o in decoded)
    return outputs


def score(hyps, references, tokenize):
    bleu = sacrebleu.corpus_bleu(hyps, [references], tokenize=tokenize).score
    chrf = sacrebleu.corpus_chrf(hyps, [references], word_order=2).score
    return bleu, chrf


def main():
    args = parse_args()
    model, tokenizer = load_model(args.base, args.adapter)

    sources, references = load_pairs(args.dataset)
    if args.limit:
        sources, references = sources[:args.limit], references[:args.limit]

    results = []
    if args.policy == "full":
        hyps = translate_full(model, tokenizer, sources, args.tgt_lang, args.batch_size)
        bleu, chrf = score(hyps, references, args.tokenize)
        results.append({"policy": "full", "k": None, "bleu": bleu, "chrf": chrf, "AL": None})
        print(f"full         BLEU={bleu:.2f}  chrF++={chrf:.2f}")
    else:
        for k in args.k:
            hyps, als = [], []
            for src in tqdm(sources, desc=f"wait-{k}"):
                hyp, trace = wait_k_decode(model, tokenizer, src, k=k,
                                           target_language=args.tgt_lang, verbose=False)
                hyps.append(hyp)
                als.append(average_lagging(trace, len(src.split())))
            bleu, chrf = score(hyps, references, args.tokenize)
            mean_al = sum(als) / len(als)
            results.append({"policy": f"wait-{k}", "k": k, "bleu": bleu,
                            "chrf": chrf, "AL": mean_al})
            print(f"wait-{k:<2}      BLEU={bleu:.2f}  chrF++={chrf:.2f}  AL={mean_al:.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "adapter": args.adapter,
                       "tokenize": args.tokenize, "results": results}, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()

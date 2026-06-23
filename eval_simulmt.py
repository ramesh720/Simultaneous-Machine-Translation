"""Multi-GPU simultaneous-MT evaluation: full-sentence vs wait-k decoding,
over IN22-Conv / IN22-Gen for several Indian<->English directions.

Efficiency: data-parallel replication via Accelerate. Each process loads a full
copy of the model on its own GPU and decodes a disjoint shard of the sentences;
the main process gathers all hypotheses, restores order, and scores
BLEU (sacreBLEU, `intl` tokenizer) + mean Average Lagging (AL, wait-k only).

Run on 4 GPUs:
    accelerate launch --num_processes 4 eval_simulmt.py \
        --data_dir eval_data --out_dir eval_results \
        --langs te hi gu ta --datasets conv gen --directions x2e e2x \
        --policies full waitk --k 1 3 5 7

Single GPU (debug / small subset):
    python eval_simulmt.py --data_dir eval_data --langs te --datasets conv \
        --directions x2e --policies full waitk --k 3 --limit 20
"""
import argparse
import json
import sys
from pathlib import Path

import sacrebleu
import torch
from tqdm.auto import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "waitk_finetune"))
from src.load import load_model                                    # noqa: E402
from src.waitk import average_lagging, build_prompt, load_pairs, wait_k_decode  # noqa: E402

LANG_NAME = {"te": "Telugu", "hi": "Hindi", "gu": "Gujarati",
             "ta": "Tamil", "en": "English"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="sarvamai/sarvam-translate")
    p.add_argument("--adapter", default=None, help="optional LoRA adapter / fine-tuned dir")
    p.add_argument("--data_dir", default="eval_data")
    p.add_argument("--out_dir", default="eval_results")
    p.add_argument("--langs", nargs="+", default=["te", "hi", "gu", "ta"])
    p.add_argument("--datasets", nargs="+", default=["conv", "gen"], choices=["conv", "gen"])
    p.add_argument("--directions", nargs="+", default=["x2e", "e2x"],
                   choices=["x2e", "e2x"], help="x2e: Indic->English, e2x: English->Indic")
    p.add_argument("--policies", nargs="+", default=["full", "waitk"], choices=["full", "waitk"])
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--batch-size", type=int, default=8, help="full-sentence batch size (per GPU)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--tokenize", default="intl", help="sacrebleu tokenizer (intl matches IN22)")
    p.add_argument("--limit", type=int, default=None, help="evaluate first N sentences")
    return p.parse_args()


def set_spec(ds, direction, lc):
    """Return (json filename stem, target-language name) for a (dataset, dir, lang)."""
    if direction == "x2e":
        stem = f"in22_{ds}_{lc}_en"
        tgt_lang = "English"
    else:                                   # e2x
        stem = f"in22_{ds}_en_{lc}"
        tgt_lang = LANG_NAME[lc]
    return stem, tgt_lang


@torch.inference_mode()
def full_shard(model, tokenizer, shard, tgt_lang, batch_size, max_new_tokens):
    """shard: list of (idx, src). Length-sorted batching to cut padding waste."""
    order = sorted(range(len(shard)), key=lambda i: len(shard[i][1]))
    out = []
    for s in range(0, len(order), batch_size):
        chunk = [shard[order[j]] for j in range(s, min(s + batch_size, len(order)))]
        texts = [build_prompt(tokenizer, t, tgt_lang) for _, t in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=tokenizer.pad_token_id)
        gen = gen[:, enc.input_ids.shape[1]:]
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for (idx, _), o in zip(chunk, dec):
            out.append((idx, o.split("<end_of_turn>")[0].strip(), None))
    return out


def waitk_shard(model, tokenizer, shard, tgt_lang, k, max_new_tokens, show):
    out = []
    for idx, src in tqdm(shard, desc=f"wait-{k}", disable=not show):
        hyp, trace = wait_k_decode(model, tokenizer, src, k=k, target_language=tgt_lang,
                                   max_target_tokens=max_new_tokens, verbose=False)
        out.append((idx, hyp, average_lagging(trace, len(src.split()))))
    return out


def evaluate_policy(model, tokenizer, state, sources, references, tgt_lang,
                    policy, k, args):
    """Run one policy over one set across all GPUs; main process returns (row, records)."""
    indexed = list(enumerate(sources))
    shard = indexed[state.process_index::state.num_processes]   # interleaved, balanced

    if policy == "full":
        local = full_shard(model, tokenizer, shard, tgt_lang, args.batch_size,
                           args.max_new_tokens)
        tag = "full"
    else:
        local = waitk_shard(model, tokenizer, shard, tgt_lang, k, args.max_new_tokens,
                            show=state.is_main_process)
        tag = f"wait-{k}"

    gathered = gather_object(local)
    state.wait_for_everyone()
    if not state.is_main_process:
        return None, None

    gathered.sort(key=lambda r: r[0])
    hyps = [h for _, h, _ in gathered]
    als = [a for _, _, a in gathered if a is not None]
    bleu = sacrebleu.corpus_bleu(hyps, [references], tokenize=args.tokenize).score
    mean_al = (sum(als) / len(als)) if als else None

    records = [{"input": sources[i], "reference": references[i], "hypothesis": h,
                "AL": a} for (i, h, a) in gathered]
    row = {"policy": tag, "k": (None if policy == "full" else k),
           "bleu": bleu, "AL": mean_al, "n": len(hyps)}
    return row, records


def main():
    args = parse_args()
    state = PartialState()
    out_dir = Path(args.out_dir)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)

    if state.is_main_process:
        print(f"Loading {args.base} on {state.num_processes} GPU(s) ...")
    model, tokenizer = load_model(args.base, args.adapter, device=str(state.device))

    summary = []
    for ds in args.datasets:
        for direction in args.directions:
            for lc in args.langs:
                stem, tgt_lang = set_spec(ds, direction, lc)
                path = Path(args.data_dir) / f"{stem}.json"
                if not path.exists():
                    if state.is_main_process:
                        print(f"[skip] missing {path} (run prepare_in22_data.py first)")
                    continue

                sources, references = load_pairs(str(path))
                if args.limit:
                    sources, references = sources[:args.limit], references[:args.limit]
                if state.is_main_process:
                    print(f"\n=== {stem}  ({len(sources)} sents, target={tgt_lang}) ===")

                # build the list of policy runs: full once, wait-k per k
                runs = []
                if "full" in args.policies:
                    runs.append(("full", None))
                if "waitk" in args.policies:
                    runs += [("waitk", k) for k in args.k]

                for policy, k in runs:
                    row, records = evaluate_policy(model, tokenizer, state, sources,
                                                   references, tgt_lang, policy, k, args)
                    if not state.is_main_process:
                        continue
                    tag = row["policy"]
                    al_str = f"  AL={row['AL']:.2f}" if row["AL"] is not None else ""
                    print(f"  {tag:<10} BLEU={row['bleu']:.2f}{al_str}")

                    out_path = out_dir / "outputs" / f"{stem}_{tag}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"dataset": ds, "direction": direction, "lang": lc,
                                   "target_language": tgt_lang, "policy": tag,
                                   "bleu": row["bleu"], "AL": row["AL"],
                                   "records": records}, f, indent=2, ensure_ascii=False)

                    summary.append({"dataset": ds, "direction": direction, "lang": lc,
                                    **row})

    if state.is_main_process:
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump({"base": args.base, "adapter": args.adapter,
                       "tokenize": args.tokenize, "results": summary}, f, indent=2)
        print(f"\nSaved {len(summary)} rows -> {out_dir / 'metrics_summary.json'}")
        _print_table(summary)


def _print_table(summary):
    print("\n" + "=" * 64)
    print(f"{'dataset':<6}{'dir':<5}{'lang':<5}{'policy':<10}{'BLEU':>8}{'AL':>8}")
    print("-" * 64)
    for r in summary:
        al = f"{r['AL']:.2f}" if r["AL"] is not None else "-"
        print(f"{r['dataset']:<6}{r['direction']:<5}{r['lang']:<5}"
              f"{r['policy']:<10}{r['bleu']:>8.2f}{al:>8}")


if __name__ == "__main__":
    main()

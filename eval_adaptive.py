"""Multi-GPU simultaneous-MT evaluation with ADAPTIVE read/write policies.

Extends eval_simulmt.py with two adaptive decoders (no extra training needed):

    la   : Local Agreement (LA-n)  -- commit a token once the last n
           re-translations agree on it.  Sweep with --n.
    conf : Confidence threshold     -- WRITE the next greedy token only when its
           probability >= tau, else READ more source.  Sweep with --tau.

Also keeps `full` and `waitk` so every policy lands on the SAME BLEU-vs-AL axes.

Efficiency: Accelerate data-parallel replication (one full model copy per GPU,
disjoint sentence shards), gathered + scored on the main process.
BLEU = sacreBLEU (`intl` tokenizer, matches IN22); AL = mean Average Lagging.

Run on 4 GPUs:
    accelerate launch --num_processes 4 eval_adaptive.py \
        --data_dir eval_data --out_dir eval_results_adaptive \
        --langs te hi gu ta --datasets conv gen --directions x2e e2x \
        --policies full waitk la conf --k 1 3 5 7 --n 2 3 --tau 0.4 0.6 0.8

Quick smoke test:
    accelerate launch --num_processes 4 eval_adaptive.py \
        --langs te --datasets conv --directions x2e \
        --policies la conf --n 2 --tau 0.6 --limit 40
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
from src.load import load_model                                          # noqa: E402
from src.waitk import average_lagging, laal, build_prompt, load_pairs, wait_k_decode  # noqa: E402
from src.adaptive import local_agreement_decode, confidence_decode       # noqa: E402
from src.comet_eval import score_comet  # noqa: E402

LANG_NAME = {"te": "Telugu", "hi": "Hindi", "gu": "Gujarati",
             "ta": "Tamil", "en": "English"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="sarvamai/sarvam-translate")
    p.add_argument("--adapter", default=None, help="optional LoRA adapter / fine-tuned dir")
    p.add_argument("--data_dir", default="eval_data")
    p.add_argument("--out_dir", default="eval_results_adaptive")
    p.add_argument("--langs", nargs="+", default=["te", "hi", "gu", "ta"])
    p.add_argument("--datasets", nargs="+", default=["conv", "gen"], choices=["conv", "gen"])
    p.add_argument("--directions", nargs="+", default=["x2e", "e2x"],
                   choices=["x2e", "e2x"], help="x2e: Indic->English, e2x: English->Indic")
    p.add_argument("--policies", nargs="+", default=["full", "waitk", "la", "conf"],
                   choices=["full", "waitk", "la", "conf"])
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 7], help="wait-k values")
    p.add_argument("--n", type=int, nargs="+", default=[2], help="local-agreement depths")
    p.add_argument("--tau", type=float, nargs="+", default=[0.6], help="confidence thresholds")
    p.add_argument("--batch-size", type=int, default=8, help="full-sentence batch size (per GPU)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--tokenize", default="intl", help="sacrebleu tokenizer (intl matches IN22)")
    p.add_argument("--limit", type=int, default=None, help="evaluate first N sentences")
    return p.parse_args()


def set_spec(ds, direction, lc):
    """Return (json filename stem, target-language name) for a (dataset, dir, lang)."""
    if direction == "x2e":
        return f"in22_{ds}_{lc}_en", "English"
    return f"in22_{ds}_en_{lc}", LANG_NAME[lc]


def expand_runs(policies, args):
    """Flatten the requested policies into (policy, param, tag) triples."""
    runs = []
    if "full" in policies:
        runs.append(("full", None, "full"))
    if "waitk" in policies:
        runs += [("waitk", k, f"wait-{k}") for k in args.k]
    if "la" in policies:
        runs += [("la", n, f"la-{n}") for n in args.n]
    if "conf" in policies:
        runs += [("conf", tau, f"conf-{tau}") for tau in args.tau]
    return runs


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


def streaming_shard(model, tokenizer, shard, tgt_lang, policy, param, max_new_tokens, show):
    """Per-sentence streaming decode for waitk / la / conf. Returns (idx, hyp, AL)."""
    out = []
    for idx, src in tqdm(shard, desc=policy, disable=not show):
        if policy == "waitk":
            hyp, trace = wait_k_decode(model, tokenizer, src, k=param,
                                       target_language=tgt_lang,
                                       max_target_tokens=max_new_tokens, verbose=False)
        elif policy == "la":
            hyp, trace = local_agreement_decode(model, tokenizer, src, n=param,
                                                target_language=tgt_lang,
                                                max_target_tokens=max_new_tokens, verbose=False)
        else:  # conf
            hyp, trace = confidence_decode(model, tokenizer, src, tau=param,
                                           target_language=tgt_lang,
                                           max_target_tokens=max_new_tokens, verbose=False)
        out.append((idx, hyp, average_lagging(trace, len(src.split())),
                   laal(trace, len(src.split()))))
    return out


def evaluate_run(model, tokenizer, state, sources, references, tgt_lang,
                 policy, param, args):
    """Run one (policy, param) over one set across all GPUs; main returns (row, records)."""
    indexed = list(enumerate(sources))
    shard = indexed[state.process_index::state.num_processes]   # interleaved, balanced

    if policy == "full":
        local = full_shard(model, tokenizer, shard, tgt_lang, args.batch_size,
                           args.max_new_tokens)
    else:
        local = streaming_shard(model, tokenizer, shard, tgt_lang, policy, param,
                               args.max_new_tokens, show=state.is_main_process)

    gathered = gather_object(local)
    state.wait_for_everyone()
    if not state.is_main_process:
        return None, None

    gathered.sort(key=lambda r: r[0])
    hyps = [h for _, h, *_ in gathered]
    als = [r[2] for r in gathered if r[2] is not None]
    laals = [r[3] for r in gathered if len(r) > 3 and r[3] is not None]
    bleu = sacrebleu.corpus_bleu(hyps, [references], tokenize=args.tokenize).score
    mean_al = (sum(als) / len(als)) if als else None
    mean_laal = (sum(laals) / len(laals)) if laals else None

    # COMET scoring (on main process only)
    comet_score = None
    try:
        comet_result = score_comet(sources, hyps, references, gpus=1)
        if comet_result:
            comet_score = comet_result["system_score"]
    except Exception as e:
        print(f"  [comet] skipped: {e}")

    records = [{"input": sources[i], "reference": references[i], "hypothesis": h,
                "AL": r[2], "LAAL": r[3] if len(r) > 3 else None}
               for (r, h) in zip(gathered, hyps)]
    row = {"policy": policy, "param": param, "bleu": bleu, "comet": comet_score,
           "AL": mean_al, "LAAL": mean_laal, "n": len(hyps)}
    return row, records


def main():
    args = parse_args()
    state = PartialState()
    out_dir = Path(args.out_dir)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)

    if state.is_main_process:
        print(f"Loading {args.base} on {state.num_processes} GPU(s) ...")
    model, tokenizer = load_model(args.base, args.adapter, device=str(state.device))

    runs = expand_runs(args.policies, args)
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

                for policy, param, tag in runs:
                    row, records = evaluate_run(model, tokenizer, state, sources,
                                                references, tgt_lang, policy, param, args)
                    if not state.is_main_process:
                        continue
                    al_str = f"  AL={row['AL']:.2f}" if row["AL"] is not None else ""
                    print(f"  {tag:<10} BLEU={row['bleu']:.2f}{al_str}")

                    with open(out_dir / "outputs" / f"{stem}_{tag}.json", "w",
                              encoding="utf-8") as f:
                        json.dump({"dataset": ds, "direction": direction, "lang": lc,
                                   "target_language": tgt_lang, "policy": tag,
                                   "bleu": row["bleu"], "AL": row["AL"],
                                   "records": records}, f, indent=2, ensure_ascii=False)

                    summary.append({"dataset": ds, "direction": direction, "lang": lc,
                                    "policy": tag, "family": policy, "param": param,
                                    "bleu": row["bleu"], "AL": row["AL"], "n": row["n"]})

    if state.is_main_process:
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump({"base": args.base, "adapter": args.adapter,
                       "tokenize": args.tokenize, "results": summary}, f, indent=2)
        print(f"\nSaved {len(summary)} rows -> {out_dir / 'metrics_summary.json'}")
        _print_table(summary)


def _print_table(summary):
    print("\n" + "=" * 70)
    print(f"{'dataset':<6}{'dir':<5}{'lang':<5}{'policy':<12}{'BLEU':>8}{'AL':>8}")
    print("-" * 70)
    for r in summary:
        al = f"{r['AL']:.2f}" if r["AL"] is not None else "-"
        print(f"{r['dataset']:<6}{r['direction']:<5}{r['lang']:<5}"
              f"{r['policy']:<12}{r['bleu']:>8.2f}{al:>8}")


if __name__ == "__main__":
    main()

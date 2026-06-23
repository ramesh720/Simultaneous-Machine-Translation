"""Plot the BLEU-vs-Average-Lagging frontier per language/direction from a
metrics_summary.json written by eval_adaptive.py (or eval_simulmt.py).

Each policy family (wait-k, la, conf) is drawn as a connected curve sorted by
latency; `full` is shown as a horizontal "offline ceiling" line. One figure per
dataset, with a subplot grid of languages x directions.

Usage:
    python plot_results.py --summary eval_results_adaptive/metrics_summary.json
    python plot_results.py --summary eval_results_adaptive/metrics_summary.json \
        --out_dir eval_results_adaptive/plots
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

LANG_NAME = {"te": "Telugu", "hi": "Hindi", "gu": "Gujarati", "ta": "Tamil"}
DIR_NAME = {"x2e": "Indic→English", "e2x": "English→Indic"}
STYLE = {                          # family -> (label, color, marker)
    "wait": ("wait-k", "#1f77b4", "o"),
    "la":   ("local-agreement", "#2ca02c", "^"),
    "conf": ("confidence", "#d62728", "s"),
}


def family_of(row):
    """Return (family_key, param_label) for a result row, tolerating both
    eval_adaptive (has 'family'/'param') and eval_simulmt (has 'k') formats."""
    fam = row.get("family")
    policy = row.get("policy", "")
    if fam is None:                       # derive from the policy tag
        fam = "full" if policy == "full" else policy.split("-")[0]
    param = row.get("param", row.get("k"))
    if param is None:
        m = re.search(r"-([0-9.]+)$", policy)
        param = m.group(1) if m else ""
    return fam, str(param)


def group_key(r):
    return (r["dataset"], r["direction"], r["lang"])


def plot_axis(ax, rows, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Average Lagging (AL)")
    ax.set_ylabel("BLEU")
    ax.grid(True, alpha=0.3)

    by_family = defaultdict(list)         # family -> [(AL, bleu, param)]
    ceiling = None
    for r in rows:
        fam, param = family_of(r)
        if fam == "full":
            ceiling = r["bleu"]
            continue
        if r.get("AL") is None:
            continue
        by_family[fam].append((r["AL"], r["bleu"], param))

    if ceiling is not None:
        ax.axhline(ceiling, ls="--", color="gray", lw=1.2,
                   label=f"full (offline) = {ceiling:.1f}")

    for fam, pts in by_family.items():
        pts.sort(key=lambda t: t[0])
        label, color, marker = STYLE.get(fam, (fam, None, "x"))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=marker, color=color, label=label, lw=1.6, ms=6)
        for al, bleu, param in pts:       # annotate each point with its setting
            ax.annotate(str(param), (al, bleu), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=color)

    if by_family or ceiling is not None:
        ax.legend(fontsize=7, loc="lower right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="metrics_summary.json")
    ap.add_argument("--out_dir", default=None, help="default: <summary dir>/plots")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    with open(summary_path, encoding="utf-8") as f:
        results = json.load(f)["results"]
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for r in results:
        grouped[group_key(r)].append(r)

    datasets = sorted({k[0] for k in grouped})
    for ds in datasets:
        langs = sorted({k[2] for k in grouped if k[0] == ds})
        dirs = sorted({k[1] for k in grouped if k[0] == ds})
        nrows, ncols = len(langs), len(dirs)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                                 squeeze=False)
        for i, lang in enumerate(langs):
            for j, direction in enumerate(dirs):
                rows = grouped.get((ds, direction, lang), [])
                title = f"IN22-{ds}  {LANG_NAME.get(lang, lang)}  " \
                        f"{DIR_NAME.get(direction, direction)}"
                if rows:
                    plot_axis(axes[i][j], rows, title)
                else:
                    axes[i][j].set_visible(False)
        fig.suptitle(f"Quality vs Latency — IN22-{ds}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = out_dir / f"frontier_{ds}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

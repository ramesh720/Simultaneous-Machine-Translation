"""
Build IN22 test sets (both directions) for several Indian languages, in the JSON
format used by the wait-k evaluator.

For every (benchmark, language) it writes two files, one per direction:

    in22_{ds}_{lc}_en.json   ->  {"input": "<Indic>",  "reference": "<English>"}   (X -> En)
    in22_{ds}_en_{lc}.json   ->  {"input": "<English>", "reference": "<Indic>"}    (En -> X)

where ds in {conv, gen} and lc in {te, hi, gu, ta}.

Usage:
    python prepare_in22_data.py --out_dir ./eval_data
    python prepare_in22_data.py --out_dir ./eval_data --langs te,hi
"""
import os
import json
import fire
from datasets import load_dataset

ENG_CODE = "eng_Latn"

# short code -> FLORES-style IN22 column code
LANGS = {
    "te": "tel_Telu",   # Telugu
    "hi": "hin_Deva",   # Hindi
    "gu": "guj_Gujr",   # Gujarati
    "ta": "tam_Taml",   # Tamil
}

DATASETS = {
    "gen": "ai4bharat/IN22-Gen",
    "conv": "ai4bharat/IN22-Conv",
}


def _load_split(dataset_name):
    """Load the dataset's `default` config (all languages as columns)."""
    ds = load_dataset(dataset_name, "default", token=True)
    split = "gen" if "gen" in ds else ("conv" if "conv" in ds else list(ds.keys())[0])
    return ds[split]


def _write(records, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, ensure_ascii=False)
    print(f"  wrote {len(records):5d} examples -> {path}")


def main(out_dir: str = "./eval_data", langs: str = "te,hi,gu,ta"):
    os.makedirs(out_dir, exist_ok=True)
    print(langs)
    selected = [l.strip() for l in langs if l.strip()]
    for lc in selected:
        if lc not in LANGS:
            raise ValueError(f"Unknown language code '{lc}'. Choose from {list(LANGS)}.")

    for ds_key, ds_name in DATASETS.items():
        print(f"Loading {ds_name} ...")
        split = _load_split(ds_name)

        for lc in selected:
            code = LANGS[lc]
            x2e, e2x = [], []
            for row in split:
                indic = row.get(code)
                eng = row.get(ENG_CODE)
                if indic is None or eng is None:
                    raise KeyError(
                        f"Columns '{code}'/'{ENG_CODE}' not found; got {list(row.keys())}"
                    )
                indic, eng = indic.strip(), eng.strip()
                x2e.append({"input": indic, "reference": eng})
                e2x.append({"input": eng, "reference": indic})

            _write(x2e, os.path.join(out_dir, f"in22_{ds_key}_{lc}_en.json"))
            _write(e2x, os.path.join(out_dir, f"in22_{ds_key}_en_{lc}.json"))


if __name__ == "__main__":
    fire.Fire(main)

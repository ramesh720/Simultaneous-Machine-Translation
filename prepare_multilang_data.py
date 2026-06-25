"""Download and merge parallel corpora for multi-language wait-k training.

Builds a single TSV with columns: src_lang, tgt_lang, src, tgt
covering all requested language pairs in both directions (X<->En).

Uses the Samanantar dataset from ai4bharat on HuggingFace, falling back
to BPCC if Samanantar is unavailable.

Usage:
    python prepare_multilang_data.py --out_path multilang_train.tsv
    python prepare_multilang_data.py --out_path multilang_train.tsv --langs te,hi --max_per_direction 20000
"""
import os
import random
import fire
import pandas as pd

# Language code -> (FLORES code, full name, Samanantar HF config)
LANG_INFO = {
    "te": ("tel_Telu", "Telugu",    "en-te"),
    "hi": ("hin_Deva", "Hindi",     "en-hi"),
    "gu": ("guj_Gujr", "Gujarati",  "en-gu"),
    "ta": ("tam_Taml", "Tamil",     "en-ta"),
}


def _load_samanantar(lang_code: str, max_pairs: int):
    """Load Samanantar parallel corpus for a language pair."""
    from datasets import load_dataset

    _, _, config = LANG_INFO[lang_code]

    try:
        ds = load_dataset("ai4bharat/samanantar", config, split="train",
                          trust_remote_code=True, token=True)
        pairs = []
        for row in ds:
            src = row.get("src", "").strip()
            tgt = row.get("tgt", "").strip()
            if src and tgt:
                pairs.append((src, tgt))  # (English, Indic)
            if len(pairs) >= max_pairs * 2:
                break
        return pairs
    except Exception as e:
        print(f"  [warn] Samanantar failed for {lang_code}: {e}")
        return _load_bpcc_fallback(lang_code, max_pairs)


def _load_bpcc_fallback(lang_code: str, max_pairs: int):
    """Fallback: load BPCC or local TSV data."""
    flores_code = LANG_INFO[lang_code][0]
    tsv_path = f"{flores_code}.tsv"

    # Try local TSV first
    for candidate in [tsv_path, f"{flores_code}_samantar.tsv",
                      f"../{flores_code}.tsv", f"../{flores_code}_samantar.tsv"]:
        if os.path.exists(candidate):
            print(f"  [fallback] Loading from local file: {candidate}")
            df = pd.read_csv(candidate, sep="\t").dropna(subset=["src", "tgt"])
            pairs = list(zip(df["src"].astype(str), df["tgt"].astype(str)))
            return pairs[:max_pairs * 2]

    # Try BPCC from HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/BPCC", split="train", token=True)
        pairs = []
        for row in ds:
            if row.get("langpair", "") == f"en-{lang_code}":
                src = row.get("src", "").strip()
                tgt = row.get("tgt", "").strip()
                if src and tgt:
                    pairs.append((src, tgt))
                if len(pairs) >= max_pairs * 2:
                    break
        return pairs
    except Exception as e:
        print(f"  [warn] BPCC fallback also failed for {lang_code}: {e}")
        return []


def main(
    out_path: str = "multilang_train.tsv",
    langs: str = "te,hi,gu,ta",
    max_per_direction: int = 50000,
    seed: int = 42,
):
    """Download and merge training data for all specified languages."""
    random.seed(seed)
    selected = [l.strip() for l in langs.split(",") if l.strip()]

    all_rows = []
    for lc in selected:
        if lc not in LANG_INFO:
            raise ValueError(f"Unknown language code '{lc}'. Choose from {list(LANG_INFO)}")

        flores_code, lang_name, _ = LANG_INFO[lc]
        print(f"\nLoading data for {lang_name} ({lc})...")

        pairs = _load_samanantar(lc, max_per_direction)
        if not pairs:
            print(f"  [skip] No data found for {lang_name}")
            continue

        random.shuffle(pairs)

        # X -> English direction
        x2e_pairs = pairs[:max_per_direction]
        for en_text, indic_text in x2e_pairs:
            all_rows.append({
                "src_lang": lang_name,
                "tgt_lang": "English",
                "src": indic_text,    # Indic source
                "tgt": en_text,       # English target
            })

        # English -> X direction
        e2x_pairs = pairs[:max_per_direction]
        for en_text, indic_text in e2x_pairs:
            all_rows.append({
                "src_lang": "English",
                "tgt_lang": lang_name,
                "src": en_text,       # English source
                "tgt": indic_text,    # Indic target
            })

        print(f"  {lang_name}: {len(x2e_pairs)} x2e + {len(e2x_pairs)} e2x pairs")

    if not all_rows:
        print("\nERROR: No data loaded for any language. Check HuggingFace access.")
        return

    # Shuffle all rows together
    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {len(df)} total rows -> {out_path}")
    print(f"Languages: {df['src_lang'].unique().tolist()}")
    print(f"Directions: {df.groupby(['src_lang', 'tgt_lang']).size().to_dict()}")


if __name__ == "__main__":
    fire.Fire(main)

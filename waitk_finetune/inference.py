"""Translate a single sentence with full-sentence or wait-k decoding.

Examples:
    python inference.py --text "నేను రోజూ ఉదయం పార్కులో నడుస్తాను." --policy waitk --k 3 --trace
    python inference.py --text "..." --policy full --adapter checkpoints/final
"""
import argparse

import torch

from src.load import load_model
from src.waitk import build_prompt, wait_k_decode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="source sentence")
    p.add_argument("--base", default="sarvamai/sarvam-translate")
    p.add_argument("--adapter", default=None)
    p.add_argument("--policy", choices=["full", "waitk"], default="waitk")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--tgt-lang", default="English")
    p.add_argument("--trace", action="store_true", help="print READ/WRITE trace (wait-k)")
    return p.parse_args()


@torch.inference_mode()
def translate_full(model, tokenizer, text, tgt_lang):
    prompt = build_prompt(tokenizer, text, tgt_lang)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**enc, max_new_tokens=256, do_sample=False, num_beams=1,
                         pad_token_id=tokenizer.pad_token_id)
    out = tokenizer.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out.split("<end_of_turn>")[0].strip()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.base, args.adapter)

    if args.policy == "full":
        print(translate_full(model, tokenizer, args.text, args.tgt_lang))
    else:
        translation, _ = wait_k_decode(model, tokenizer, args.text, k=args.k,
                                       target_language=args.tgt_lang, verbose=args.trace)
        if not args.trace:
            print(translation)


if __name__ == "__main__":
    main()

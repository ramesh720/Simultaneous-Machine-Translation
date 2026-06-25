"""Shared wait-k utilities: prompt building, prefix construction,
simultaneous decoding, and the Average Lagging latency metric.

These functions are used by both the data pipeline (to build wait-k training
prefixes) and the evaluation / inference scripts (to decode under a wait-k
policy). Keeping them in one place guarantees train- and test-time use the
exact same prompt format.
"""
from __future__ import annotations

import json
from typing import List, Tuple

import torch


def build_prompt(tokenizer, source_text: str, target_language: str) -> str:
    """Render the sarvam-translate chat prompt for a (partial) source."""
    messages = [
        {"role": "system", "content": f"Translate the text below to {target_language}."},
        {"role": "user", "content": source_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def waitk_prefix(
    source_words: List[str], target_words: List[str], k: int, j: int
) -> Tuple[str, int]:
    """Given a source prefix of ``j`` words, return the wait-k aligned
    (source_prefix_text, num_target_words).

    Wait-k: target token ``i`` (1-indexed) is emitted after reading
    ``k + i - 1`` source words. Inverting, after reading ``j`` source words
    the model should have produced ``i = j - k + 1`` target tokens.
    """
    i = max(0, j - k + 1)
    i = min(i, len(target_words))
    src_prefix = " ".join(source_words[:j])
    return src_prefix, i


def stop_token_ids(tokenizer) -> List[int]:
    """EOS plus any chat end-of-turn tokens (gemma3 uses <end_of_turn>)."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for tok in ["<end_of_turn>", "<|im_end|>", "<eos>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


@torch.inference_mode()
def wait_k_decode(
    model,
    tokenizer,
    source: str,
    k: int = 3,
    target_language: str = "English",
    max_target_tokens: int = 256,
    verbose: bool = False,
):
    """Simultaneous wait-k decoding for a single sentence.

    Re-prompts the model with a growing source prefix and reads off one greedy
    token per step. EOS is suppressed until the full source has been read.
    Returns ``(translation, trace)`` where trace is a list of
    ``("READ"|"WRITE"|"STOP", detail)`` tuples.
    """
    eos_ids = stop_token_ids(tokenizer)
    src_words = source.split()
    S = len(src_words)

    committed: List[int] = []
    trace: List[Tuple[str, str]] = []
    prev_read = 0

    for t in range(1, max_target_tokens + 1):
        num_src = min(k + t - 1, S)
        while prev_read < num_src:
            prev_read += 1
            trace.append(("READ", src_words[prev_read - 1]))

        prompt = build_prompt(tokenizer, " ".join(src_words[:num_src]), target_language)
        first_device = next(model.parameters()).device
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(first_device)
        if committed:
            tail = torch.tensor([committed], device=first_device, dtype=prompt_ids.dtype)
            input_ids = torch.cat([prompt_ids, tail], dim=1)
        else:
            input_ids = prompt_ids

        logits = model(input_ids=input_ids).logits[0, -1]
        if num_src < S:                       # cannot finish before reading all source
            for eid in eos_ids:
                logits[eid] = float("-inf")

        next_id = int(logits.argmax())
        if next_id in eos_ids:
            trace.append(("STOP", "<eos>"))
            break

        committed.append(next_id)
        trace.append(("WRITE", tokenizer.decode([next_id], skip_special_tokens=True)))

    translation = tokenizer.decode(committed, skip_special_tokens=True).strip()

    if verbose:
        _print_trace(source, k, S, trace, translation)
    return translation, trace


def _print_trace(source, k, S, trace, translation):
    print(f"Source : {source}")
    print(f"Policy : wait-{k}  (source words = {S})\n")
    read_buf, tgt_buf = [], ""
    for action, detail in trace:
        if action == "READ":
            read_buf.append(detail)
            print(f"  READ  -> [{' '.join(read_buf)}]")
        elif action == "WRITE":
            tgt_buf += detail
            print(f"  WRITE -> {detail!r:<12} (target so far:{tgt_buf})")
        else:
            print(f"  {action}")
    print(f"\nFinal translation: {translation}")


def average_lagging(trace, num_src_words: int) -> float:
    """Average Lagging (Ma et al., 2019) from a READ/WRITE action trace."""
    src_read, g = 0, []
    for action, _ in trace:
        if action == "READ":
            src_read += 1
        elif action == "WRITE":
            g.append(src_read)
    if not g:
        return float("nan")
    tgt_len, src_len = len(g), num_src_words
    rate = tgt_len / src_len
    tau = next((i + 1 for i, gv in enumerate(g) if gv >= src_len), tgt_len)
    return sum(g[t] - t / rate for t in range(tau)) / tau


def laal(trace, num_src_words: int) -> float:
    """Length-Adaptive Average Lagging — unbiased for over/under-generation.

    Standard AL underestimates latency when the system over-generates.
    LAAL normalises by max(|target|, |source|) instead of tau, ensuring
    fair comparison across systems with different output lengths.
    """
    src_read, g = 0, []
    for action, _ in trace:
        if action == "READ":
            src_read += 1
        elif action == "WRITE":
            g.append(src_read)
    if not g:
        return float("nan")
    tgt_len, src_len = len(g), num_src_words
    rate = tgt_len / src_len if src_len > 0 else 1.0
    tau_cutoff = next((i + 1 for i, gv in enumerate(g) if gv >= src_len), tgt_len)
    denominator = max(tgt_len, src_len)  # Length-adaptive normalisation
    return sum(g[t] - t / rate for t in range(tau_cutoff)) / denominator


def load_pairs(path: str):
    """Load an IN22-style json: returns (sources, references)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d["input"] for d in data], [d["reference"] for d in data]

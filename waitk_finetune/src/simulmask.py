"""SimulMask: Attention-mask-based wait-k training (EMNLP 2024).

Instead of truncating the source for wait-k training (data augmentation),
SimulMask constrains the model's **attention mask** so that each target
token can only attend to the wait-k-aligned source prefix.

Advantages over prefix truncation:
  - No data modification needed — full source/target pairs are used.
  - Better KV-cache efficiency at inference time.
  - Less train/inference mismatch.
  - Model retains full-sentence context implicitly.

The mask layout for a sequence [PROMPT | SOURCE | TARGET]:
  - PROMPT tokens: fully causal (attend to all prior prompt tokens).
  - SOURCE tokens: fully causal within the prompt+source span.
  - TARGET token t: attends to all PROMPT, the first min(k+t-1, S) SOURCE
    tokens, and all prior TARGET tokens (causal).

Reference: "SimulMask: Simultaneous Translation with Attention Masking"
"""
from __future__ import annotations

from typing import List, Tuple

import torch


def word_token_boundaries(
    tokenizer, text: str, offset: int = 0
) -> List[Tuple[int, int]]:
    """Map each whitespace-delimited word to its (start_tok, end_tok) range.

    ``offset`` shifts all positions by a fixed amount (e.g. if the text is
    placed after a prompt in the token sequence).

    Returns a list of (inclusive_start, exclusive_end) token index pairs,
    one per word.
    """
    words = text.split()
    boundaries = []
    pos = offset
    for word in words:
        # Tokenize the word in isolation (no special tokens).
        toks = tokenizer(word, add_special_tokens=False).input_ids
        n = len(toks)
        boundaries.append((pos, pos + n))
        pos += n
    return boundaries


def build_simulmask(
    seq_len: int,
    prompt_len: int,
    src_boundaries: List[Tuple[int, int]],
    tgt_start: int,
    tgt_len: int,
    k: int,
    full_sentence: bool = False,
) -> torch.BoolTensor:
    """Build a 2D boolean attention mask enforcing wait-k constraints.

    Args:
        seq_len:        Total sequence length (prompt + source + target).
        prompt_len:     Number of tokens in the prompt (system + chat template).
        src_boundaries: Per-word (start, end) token indices in the source.
        tgt_start:      Token index where the target begins.
        tgt_len:        Number of target tokens.
        k:              Wait-k lag in source words.
        full_sentence:  If True, allow full attention (no masking) — used for
                        the ``full_sentence_prob`` fraction of training examples.

    Returns:
        BoolTensor of shape (seq_len, seq_len).  True = can attend.
    """
    # Start with standard causal mask (lower-triangular).
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    if full_sentence:
        return mask  # No extra masking for full-sentence examples.

    S = len(src_boundaries)  # Number of source words.

    # For each target token position, determine which source tokens are visible.
    for t_idx in range(tgt_len):
        t_pos = tgt_start + t_idx          # absolute position in the sequence
        t = t_idx + 1                       # 1-indexed target step

        # Wait-k: target token t can see source words 1..min(k+t-1, S).
        num_visible_words = min(k + t - 1, S)

        # Find the last visible source token position.
        if num_visible_words <= 0:
            # Cannot see any source — mask out entire source span.
            src_start = src_boundaries[0][0]
            src_end = src_boundaries[-1][1]
            mask[t_pos, src_start:src_end] = False
        elif num_visible_words < S:
            # Mask out source tokens beyond the visible prefix.
            invisible_start = src_boundaries[num_visible_words][0]
            invisible_end = src_boundaries[-1][1]
            mask[t_pos, invisible_start:invisible_end] = False
        # else: all source words visible, no extra masking needed.

    return mask


def build_simulmask_from_texts(
    tokenizer,
    prompt_ids: List[int],
    source_text: str,
    target_text: str,
    k: int,
    full_sentence: bool = False,
) -> torch.BoolTensor:
    """Convenience wrapper: build SimulMask from raw texts.

    Tokenizes source and target, computes word boundaries, and calls
    ``build_simulmask``.
    """
    prompt_len = len(prompt_ids)
    src_ids = tokenizer(source_text, add_special_tokens=False).input_ids
    tgt_ids = tokenizer(target_text, add_special_tokens=False).input_ids

    src_boundaries = word_token_boundaries(tokenizer, source_text, offset=prompt_len)
    tgt_start = prompt_len + len(src_ids)
    tgt_len = len(tgt_ids)
    seq_len = prompt_len + len(src_ids) + tgt_len

    return build_simulmask(
        seq_len=seq_len,
        prompt_len=prompt_len,
        src_boundaries=src_boundaries,
        tgt_start=tgt_start,
        tgt_len=tgt_len,
        k=k,
        full_sentence=full_sentence,
    )

"""LightningDataModule and Dataset for wait-k fine-tuning.

Supports two training modes:
  1. **Prefix truncation** (original): source text is truncated to simulate
     partial input under wait-k alignment.
  2. **SimulMask** (EMNLP 2024): full source/target are used, with a custom
     2D attention mask that constrains each target token to only attend to its
     wait-k-aligned source prefix.

Also supports multi-language training with per-row target language.

Loss is computed only on target tokens; the prompt is masked with -100.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from .waitk import build_prompt, waitk_prefix
from .simulmask import build_simulmask, word_token_boundaries


class WaitKDataset(Dataset):
    def __init__(self, rows, tokenizer, cfg, train: bool):
        # rows: list of (source_text, target_text) or (source_text, target_text, tgt_lang)
        self.rows = rows
        self.tok = tokenizer
        self.cfg = cfg
        self.train = train
        self.k = cfg.waitk.k
        self.full_prob = cfg.waitk.full_sentence_prob
        self.tgt_lang = cfg.data.target_language
        self.max_len = cfg.data.max_length
        self.eos_id = tokenizer.eos_token_id
        self.use_simulmask = getattr(cfg.get("simulmask", {}), "enabled", False)

    def __len__(self):
        return len(self.rows)

    def _get_target_language(self, idx: int) -> str:
        """Get target language for this row."""
        if self.tgt_lang != "auto":
            return self.tgt_lang
        # Multi-language mode: tgt_lang is third element of the row tuple
        if len(self.rows[idx]) >= 3:
            return self.rows[idx][2]
        return "English"

    def _make_example_prefix(self, source_text: str, target_text: str):
        """Original prefix-truncation method."""
        src_words = source_text.split()
        tgt_words = target_text.split()
        S = len(src_words)

        full = (not self.train) or S <= self.k or random.random() < self.full_prob
        if full:
            src_prefix, tgt_prefix = source_text, target_text
            append_eos = True
        else:
            j = random.randint(self.k, S)            # source words read
            src_prefix, i = waitk_prefix(src_words, tgt_words, self.k, j)
            tgt_prefix = " ".join(tgt_words[:i])
            append_eos = False                       # a prefix should not signal completion
        return src_prefix, tgt_prefix, append_eos

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        source_text = self.rows[idx][0]
        target_text = self.rows[idx][1]
        tgt_lang = self._get_target_language(idx)

        if self.use_simulmask:
            return self._getitem_simulmask(source_text, target_text, tgt_lang)
        else:
            return self._getitem_prefix(source_text, target_text, tgt_lang)

    def _getitem_prefix(self, source_text, target_text, tgt_lang) -> Dict:
        """Original prefix-truncation training."""
        src_prefix, tgt_prefix, append_eos = self._make_example_prefix(
            source_text, target_text
        )
        prompt = build_prompt(self.tok, src_prefix, tgt_lang)
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        target_ids = self.tok(tgt_prefix, add_special_tokens=False).input_ids
        if append_eos and self.eos_id is not None:
            target_ids = target_ids + [self.eos_id]

        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}

    def _getitem_simulmask(self, source_text, target_text, tgt_lang) -> Dict:
        """SimulMask training: full sequences with constrained attention."""
        src_words = source_text.split()
        S = len(src_words)

        # Decide if this is a full-sentence or wait-k masked example
        full = (not self.train) or S <= self.k or random.random() < self.full_prob

        prompt = build_prompt(self.tok, source_text, tgt_lang)
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        target_ids = self.tok(target_text, add_special_tokens=False).input_ids

        append_eos = True  # SimulMask always uses full target
        if append_eos and self.eos_id is not None:
            target_ids = target_ids + [self.eos_id]

        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]

        # Build SimulMask attention mask
        prompt_len = len(prompt_ids)
        src_boundaries = word_token_boundaries(self.tok, source_text, offset=0)

        # We need to find where source tokens are within the prompt.
        # The prompt contains: [system_tokens] [source_text_tokens] [gen_prompt_tokens]
        # We need to locate source text within the prompt.
        src_token_ids = self.tok(source_text, add_special_tokens=False).input_ids
        src_tok_len = len(src_token_ids)

        # Find source start position within prompt by looking for source tokens.
        # The source text is embedded within the chat template, typically after
        # "user\n" and before "\n<end_of_turn>".
        src_start_in_prompt = self._find_source_start(prompt_ids, src_token_ids)

        # Rebuild boundaries with correct offset
        src_boundaries = word_token_boundaries(
            self.tok, source_text, offset=src_start_in_prompt
        )

        tgt_start = prompt_len
        tgt_len = min(len(target_ids), self.max_len - prompt_len)
        seq_len = len(input_ids)

        mask = build_simulmask(
            seq_len=seq_len,
            prompt_len=prompt_len,
            src_boundaries=src_boundaries,
            tgt_start=tgt_start,
            tgt_len=tgt_len,
            k=self.k,
            full_sentence=full,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "simulmask": mask,
        }

    def _find_source_start(self, prompt_ids, src_token_ids):
        """Find where source tokens start within the prompt token sequence."""
        src_len = len(src_token_ids)
        for i in range(len(prompt_ids) - src_len + 1):
            if prompt_ids[i:i + src_len] == src_token_ids:
                return i
        # Fallback: assume source is roughly in the middle of the prompt
        return max(0, len(prompt_ids) // 3)


class Collator:
    def __init__(self, pad_id: int, use_simulmask: bool = False):
        self.pad_id = pad_id
        self.use_simulmask = use_simulmask

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        simulmasks = []

        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)

            if self.use_simulmask and "simulmask" in b:
                # Pad the 2D mask to (max_len, max_len)
                mask = b["simulmask"]
                seq_len = mask.shape[0]
                padded = torch.zeros(max_len, max_len, dtype=torch.bool)
                padded[:seq_len, :seq_len] = mask
                simulmasks.append(padded)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.use_simulmask and simulmasks:
            # Stack into (batch, seq, seq) and convert to float for attention
            result["simulmask"] = torch.stack(simulmasks)

        return result


class WaitKDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tok = tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        use_simulmask = getattr(cfg.get("simulmask", {}), "enabled", False)
        self.collator = Collator(pad_id, use_simulmask=use_simulmask)

    def setup(self, stage=None):
        df = pd.read_csv(self.cfg.data.tsv_path, sep="\t").dropna(
            subset=[self.cfg.data.source_column, self.cfg.data.target_column]
        )

        # Determine if we're in multi-language mode
        tgt_lang_col = getattr(self.cfg.data, "target_language_column", None)
        is_multilang = (self.cfg.data.target_language == "auto" and
                        tgt_lang_col and tgt_lang_col in df.columns)

        if is_multilang:
            rows = list(
                zip(
                    df[self.cfg.data.source_column].astype(str),
                    df[self.cfg.data.target_column].astype(str),
                    df[tgt_lang_col].astype(str),
                )
            )
        else:
            rows = list(
                zip(
                    df[self.cfg.data.source_column].astype(str),
                    df[self.cfg.data.target_column].astype(str),
                )
            )

        rng = random.Random(self.cfg.seed)
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * self.cfg.data.val_fraction))
        val_rows, train_rows = rows[:n_val], rows[n_val:]

        self.train_ds = WaitKDataset(train_rows, self.tok, self.cfg, train=True)
        self.val_ds = WaitKDataset(val_rows, self.tok, self.cfg, train=False)
        print(f"[data] train={len(self.train_ds)}  val={len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

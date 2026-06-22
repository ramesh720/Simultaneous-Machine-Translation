"""LightningDataModule and Dataset for wait-k fine-tuning.

Each training example is built on the fly: with probability
``full_sentence_prob`` we use the full (source -> target) pair, otherwise we
sample a wait-k prefix pair (source[:j] -> target[:i]) so the model learns to
translate partial input under the wait-k alignment.

Loss is computed only on target tokens; the prompt is masked with -100.
"""
from __future__ import annotations

import random
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from .waitk import build_prompt, waitk_prefix


class WaitKDataset(Dataset):
    def __init__(self, rows, tokenizer, cfg, train: bool):
        # rows: list of (source_text, target_text)
        self.rows = rows
        self.tok = tokenizer
        self.cfg = cfg
        self.train = train
        self.k = cfg.waitk.k
        self.full_prob = cfg.waitk.full_sentence_prob
        self.tgt_lang = cfg.data.target_language
        self.max_len = cfg.data.max_length
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.rows)

    def _make_example(self, source_text: str, target_text: str):
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

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        source_text, target_text = self.rows[idx]
        src_prefix, tgt_prefix, append_eos = self._make_example(source_text, target_text)

        prompt = build_prompt(self.tok, src_prefix, self.tgt_lang)
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        target_ids = self.tok(tgt_prefix, add_special_tokens=False).input_ids
        if append_eos and self.eos_id is not None:
            target_ids = target_ids + [self.eos_id]

        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class WaitKDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tok = tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self.collator = Collator(pad_id)

    def setup(self, stage=None):
        df = pd.read_csv(self.cfg.data.tsv_path, sep="\t").dropna(
            subset=[self.cfg.data.source_column, self.cfg.data.target_column]
        )
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

"""LightningModule wrapping sarvam-translate with optional LoRA."""
from __future__ import annotations

import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup


def load_base_model(cfg):
    dtype = getattr(torch, cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def apply_lora(model, cfg):
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


class WaitKLightningModule(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = load_base_model(cfg)
        if cfg.lora.enabled:
            self.model = apply_lora(self.model, cfg)
        # Saved into the checkpoint for reproducibility (model excluded).
        self.save_hyperparameters(ignore=["tokenizer"])

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.cfg.training.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def save_pretrained(self, path: str):
        """Save the (LoRA adapter or full) model + tokenizer for inference."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

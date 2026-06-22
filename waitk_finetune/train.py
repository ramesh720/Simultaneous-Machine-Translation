"""Wait-k fine-tuning entry point.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml waitk.k=5 trainer.devices=4
"""
import argparse
import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from src.data import WaitKDataModule
from src.module import WaitKLightningModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. waitk.k=5")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    pl.seed_everything(cfg.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datamodule = WaitKDataModule(cfg, tokenizer)
    module = WaitKLightningModule(cfg, tokenizer)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint.dirpath,
            filename="waitk-{epoch}-{step}-{val/loss:.3f}",
            save_top_k=cfg.checkpoint.save_top_k,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy if cfg.trainer.devices > 1 else "auto",
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.grad_accum,
        gradient_clip_val=cfg.training.max_grad_norm,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(module, datamodule=datamodule)

    # Save the final adapter/model on rank 0 for inference / evaluation.
    if trainer.is_global_zero:
        out = os.path.join(cfg.checkpoint.dirpath, "final")
        module.save_pretrained(out)
        print(f"[train] saved final model to {out}")


if __name__ == "__main__":
    main()

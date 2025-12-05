from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .config import BrainTextDataConfig, BrainTextModelConfig, BrainTextTrainingConfig
from .data import BrainTextDataModule
from .model import BrainToTextLightningModule
from .tokenizer import CharTokenizer, PhonemeTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural signal-to-text model with CTC loss")
    parser.add_argument("--data-root", type=Path, default=Path("src/data/hdf5_data_final"))
    parser.add_argument("--session-limit", type=int, default=None, help="Optionally cap number of sessions")
    parser.add_argument("--max-trials", type=int, default=None, help="Optionally limit trials per split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=9)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--default-root-dir", type=Path, default=Path("runs/brain_text"))
    parser.add_argument("--limit-train-batches", type=float, default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--time-subsample", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--use-phoneme-head", action="store_true", help="Enable auxiliary phoneme prediction head")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--dev", action="store_true", help="Enable fast development run for debugging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed)

    data_cfg = BrainTextDataConfig(
        data_root=args.data_root,
        session_limit=args.session_limit,
        max_trials_per_split=args.max_trials,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_subsample=args.time_subsample,
        max_sequence_length=args.max_seq_len,
    )
    model_cfg = BrainTextModelConfig()
    train_cfg = BrainTextTrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=args.default_root_dir,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    char_tokenizer = CharTokenizer()
    phoneme_tokenizer = PhonemeTokenizer() if args.use_phoneme_head else None

    datamodule = BrainTextDataModule(
        config=data_cfg,
        tokenizer=char_tokenizer,
        phoneme_tokenizer=phoneme_tokenizer,
    )
    module = BrainToTextLightningModule(
        data_config=data_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
        tokenizer=char_tokenizer,
        phoneme_tokenizer=phoneme_tokenizer,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_cer",
            mode="min",
            save_top_k=1,
            filename="brain-text-{epoch:02d}-{val_cer:.3f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = L.Trainer(callbacks=callbacks, **train_cfg.trainer_kwargs(), logger=WandbLogger('brain-to-text'), fast_dev_run=args.dev)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

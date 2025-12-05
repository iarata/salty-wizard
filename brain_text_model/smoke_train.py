from __future__ import annotations

import tempfile

from pathlib import Path

from lightning import pytorch as pl

from .config import BrainTextDataConfig, BrainTextModelConfig, BrainTextTrainingConfig
from .data import BrainTextDataModule, _write_dummy_hdf5
from .model import BrainToTextLightningModule
from .tokenizer import CharTokenizer


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        _write_dummy_hdf5(tmp_dir, "train")
        _write_dummy_hdf5(tmp_dir, "val")
        data_cfg = BrainTextDataConfig(
            data_root=tmp_dir,
            batch_size=2,
            num_workers=0,
            max_trials_per_split=4,
        )
        model_cfg = BrainTextModelConfig()
        train_cfg = BrainTextTrainingConfig(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            default_root_dir=tmp_dir / "logs",
        )
        tokenizer = CharTokenizer()
        datamodule = BrainTextDataModule(data_cfg, tokenizer=tokenizer)
        module = BrainToTextLightningModule(data_cfg, model_cfg, train_cfg, tokenizer=tokenizer)
        trainer = pl.Trainer(accelerator="cpu", devices=1, **train_cfg.trainer_kwargs())
        trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

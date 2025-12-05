from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class BrainTextDataConfig:
    """Settings that control how neural trials are read from disk."""

    data_root: Path = Path("src/data/hdf5_data_final")
    splits: Sequence[str] = ("train", "val")
    session_limit: int | None = None
    max_trials_per_split: int | None = None
    require_text: bool = True
    lowercase_text: bool = True
    time_subsample: int = 1
    normalize: str = "feature"  # one of {"feature", "global", "none"}
    max_sequence_length: int | None = None
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    limit_records: int | None = None
    phoneme_target_probability: float = 0.0  # optional auxiliary head

    def resolved_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve()


@dataclass(slots=True)
class BrainTextModelConfig:
    """Defines architecture hyper-parameters for the encoder-decoder stack."""

    input_dim: int = 512
    conv_channels: Sequence[int] = (256, 256, 384)
    conv_kernel_size: int = 5
    conv_stride: Sequence[int] = (2, 2, 1)
    transformer_dim: int = 384
    transformer_heads: int = 6
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    final_dropout: float = 0.1
    stochastic_depth_prob: float = 0.0

    def total_time_reduction(self) -> int:
        reduction = 1
        for stride in self.conv_stride:
            reduction *= stride
        return reduction


@dataclass(slots=True)
class BrainTextTrainingConfig:
    """Trainer defaults for the initial baseline model."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 500
    max_epochs: int = 10
    precision: str = "32-true"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    default_root_dir: Path | None = None
    log_every_n_steps: int = 50
    limit_train_batches: float | None = None
    limit_val_batches: float | None = None

    def trainer_kwargs(self) -> dict:
        return {
            "max_epochs": self.max_epochs,
            "precision": self.precision,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "gradient_clip_val": self.gradient_clip_val,
            "default_root_dir": self.default_root_dir,
            "log_every_n_steps": self.log_every_n_steps,
            "limit_train_batches": self.limit_train_batches,
            "limit_val_batches": self.limit_val_batches,
        }


def _run_config_tests() -> None:
    """Test configuration dataclasses."""
    print("[config] Running tests...")
    
    # Test BrainTextDataConfig
    data_cfg = BrainTextDataConfig()
    assert data_cfg.batch_size == 8
    assert data_cfg.splits == ("train", "val")
    assert data_cfg.normalize in {"feature", "global", "none"}
    
    # Test with custom values
    custom_cfg = BrainTextDataConfig(
        data_root=Path("/tmp/test"),
        batch_size=16,
        session_limit=5,
    )
    assert custom_cfg.batch_size == 16
    assert custom_cfg.session_limit == 5
    assert custom_cfg.resolved_root() == Path("/tmp/test").resolve()
    
    # Test BrainTextModelConfig
    model_cfg = BrainTextModelConfig()
    assert model_cfg.input_dim == 512
    assert model_cfg.transformer_heads == 6
    assert len(model_cfg.conv_channels) == len(model_cfg.conv_stride)
    
    # Test time reduction calculation
    reduction = model_cfg.total_time_reduction()
    expected = 1
    for stride in model_cfg.conv_stride:
        expected *= stride
    assert reduction == expected
    
    # Test BrainTextTrainingConfig
    train_cfg = BrainTextTrainingConfig()
    assert train_cfg.learning_rate == 3e-4
    assert train_cfg.max_epochs == 10
    
    kwargs = train_cfg.trainer_kwargs()
    assert kwargs["max_epochs"] == 10
    assert kwargs["precision"] == "32-true"
    assert "learning_rate" not in kwargs  # Not a trainer kwarg
    
    print("[config] ✓ All tests passed")


if __name__ == "__main__":
    _run_config_tests()

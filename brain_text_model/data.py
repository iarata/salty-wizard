from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import BrainTextDataConfig
from .tokenizer import CharTokenizer, PhonemeTokenizer


@dataclass(slots=True)
class TrialRecord:
    file_path: Path
    trial_key: str
    session: str
    split: str
    sentence: str
    phonemes: str | None
    n_time_steps: int


def _decode_if_bytes(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _iter_hdf5_trials(
    file_path: Path,
    session: str,
    split: str,
    require_text: bool,
) -> Iterable[TrialRecord]:
    with h5py.File(file_path, "r") as handle:
        trial_keys = [key for key in handle.keys() if key.startswith("trial_")]
        for trial_key in sorted(trial_keys):
            group = handle[trial_key]
            attrs = group.attrs
            sentence = _decode_if_bytes(attrs.get("sentence_label")) or ""
            if require_text and not sentence.strip():
                continue
            phonemes = _decode_if_bytes(attrs.get("phoneme_sequence"))
            n_time_steps = int(attrs.get("n_time_steps", group["input_features"].shape[0]))
            yield TrialRecord(
                file_path=file_path,
                trial_key=trial_key,
                session=session,
                split=split,
                sentence=sentence,
                phonemes=phonemes,
                n_time_steps=n_time_steps,
            )


def scan_hdf5_trials(
    data_root: Path,
    splits: Sequence[str],
    session_limit: int | None,
    max_trials_per_split: int | None,
    require_text: bool,
) -> Dict[str, List[TrialRecord]]:
    data_root = Path(data_root)
    session_dirs = sorted(p for p in data_root.glob("t15.*") if p.is_dir())
    if session_limit:
        session_dirs = session_dirs[:session_limit]
    per_split: Dict[str, List[TrialRecord]] = {split: [] for split in splits}
    for session_dir in session_dirs:
        session_name = session_dir.name
        for split in splits:
            if max_trials_per_split and len(per_split[split]) >= max_trials_per_split:
                continue
            h5_path = session_dir / f"data_{split}.hdf5"
            if not h5_path.exists():
                continue
            for record in _iter_hdf5_trials(h5_path, session_name, split, require_text):
                per_split[split].append(record)
                if max_trials_per_split and len(per_split[split]) >= max_trials_per_split:
                    break
    return per_split


class BrainTextDataset(Dataset):
    """Maps TrialRecord entries to tensors compatible with a CTC model."""

    def __init__(
        self,
        records: Sequence[TrialRecord],
        tokenizer: CharTokenizer,
        phoneme_tokenizer: PhonemeTokenizer | None,
        config: BrainTextDataConfig,
        shuffle_records: bool = False,
    ) -> None:
        self.records = list(records)
        if shuffle_records:
            random.shuffle(self.records)
        self.tokenizer = tokenizer
        self.phoneme_tokenizer = phoneme_tokenizer
        self.config = config
        self.normalize = config.normalize
        self.time_subsample = max(1, config.time_subsample)
        self.max_sequence_length = config.max_sequence_length

    def __len__(self) -> int:
        return len(self.records)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        if self.normalize == "feature":
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-6
            return (features - mean) / std
        if self.normalize == "global":
            mean = float(features.mean())
            std = float(features.std()) + 1e-6
            return (features - mean) / std
        return features

    def _maybe_truncate(self, features: np.ndarray) -> np.ndarray:
        if self.max_sequence_length and features.shape[0] > self.max_sequence_length:
            return features[: self.max_sequence_length]
        return features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | list[int]]:
        record = self.records[idx]
        with h5py.File(record.file_path, "r") as handle:
            data = handle[record.trial_key]["input_features"][:]
        if self.time_subsample > 1:
            data = data[:: self.time_subsample]
        data = self._maybe_truncate(data)
        data = self._normalize(data.astype(np.float32))
        tensor = torch.from_numpy(data)
        text_ids = self.tokenizer.encode(record.sentence)
        if not text_ids:
            raise ValueError(f"Record {record.trial_key} is missing transcript text")
        phoneme_ids: List[int] = []
        if self.phoneme_tokenizer and record.phonemes:
            phoneme_ids = self.phoneme_tokenizer.encode(record.phonemes)
        return {
            "features": tensor,
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text": record.sentence,
            "phoneme_ids": torch.tensor(phoneme_ids, dtype=torch.long) if phoneme_ids else None,
        }


def collate_batch(
    batch: Sequence[Dict[str, torch.Tensor | str | None]],
    pad_value: float = 0.0,
    pad_token_id: int | None = None,
):
    feature_tensors = [item["features"] for item in batch]  # type: ignore[index]
    feature_lengths = torch.tensor([tensor.shape[0] for tensor in feature_tensors], dtype=torch.long)
    padded_features = torch.nn.utils.rnn.pad_sequence(feature_tensors, batch_first=True, padding_value=pad_value)

    text_tensors = [item["text_ids"] for item in batch]  # type: ignore[index]
    target_lengths = torch.tensor([tensor.shape[0] for tensor in text_tensors], dtype=torch.long)
    targets = torch.cat(text_tensors, dim=0)

    phoneme_targets = None
    phoneme_lengths = None
    phoneme_list = [item.get("phoneme_ids") for item in batch]
    if any(t is not None and t.numel() > 0 for t in phoneme_list):
        phoneme_targets = torch.cat([t for t in phoneme_list if t is not None], dim=0)
        phoneme_lengths = torch.tensor([t.numel() if t is not None else 0 for t in phoneme_list], dtype=torch.long)

    transcripts = [item["text"] for item in batch]
    return {
        "features": padded_features,
        "feature_lengths": feature_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "transcripts": transcripts,
        "phoneme_targets": phoneme_targets,
        "phoneme_lengths": phoneme_lengths,
    }


class BrainTextDataModule(L.LightningDataModule):
    """Lightning DataModule that wraps the HDF5 copy-task corpus."""

    def __init__(
        self,
        config: BrainTextDataConfig,
        tokenizer: CharTokenizer | None = None,
        phoneme_tokenizer: PhonemeTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer or CharTokenizer()
        self.phoneme_tokenizer = phoneme_tokenizer
        self._train: BrainTextDataset | None = None
        self._val: BrainTextDataset | None = None

    def prepare_data(self) -> None:  # pragma: no cover - IO only
        # Nothing to download but we ensure the directory exists for clearer errors.
        root = self.config.resolved_root()
        if not root.exists():
            raise FileNotFoundError(f"Data root {root} does not exist")

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None and self._val is not None:
            return
        root = self.config.resolved_root()
        per_split = scan_hdf5_trials(
            data_root=root,
            splits=self.config.splits,
            session_limit=self.config.session_limit,
            max_trials_per_split=self.config.max_trials_per_split,
            require_text=self.config.require_text,
        )
        if self.phoneme_tokenizer:
            for records in per_split.values():
                for record in records:
                    if record.phonemes:
                        self.phoneme_tokenizer.maybe_add_tokens(record.phonemes)
        train_records = per_split.get("train", [])
        val_records = per_split.get("val", [])
        if not train_records:
            raise RuntimeError("No training records found. Ensure data_root has train splits.")
        if not val_records:
            val_records = train_records[-max(1, len(train_records) // 10) :]
        self._train = BrainTextDataset(
            records=train_records,
            tokenizer=self.tokenizer,
            phoneme_tokenizer=self.phoneme_tokenizer,
            config=self.config,
            shuffle_records=True,
        )
        self._val = BrainTextDataset(
            records=val_records,
            tokenizer=self.tokenizer,
            phoneme_tokenizer=self.phoneme_tokenizer,
            config=self.config,
            shuffle_records=False,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=partial(collate_batch, pad_token_id=self.tokenizer.pad_id),
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=partial(collate_batch, pad_token_id=self.tokenizer.pad_id),
        )


# ---------------------------------------------------------------------------
# Lightweight smoke test using synthetic data
# ---------------------------------------------------------------------------


def _write_dummy_hdf5(tmp_dir: Path, split: str, n_trials: int = 4) -> None:
    session_dir = tmp_dir / "t15.dummy"
    session_dir.mkdir(parents=True, exist_ok=True)
    h5_path = session_dir / f"data_{split}.hdf5"
    with h5py.File(h5_path, "w") as handle:
        for idx in range(n_trials):
            group = handle.create_group(f"trial_{idx:05d}")
            data = np.random.randn(200, 512).astype(np.float32)
            group.create_dataset("input_features", data=data)
            group.attrs["sentence_label"] = f"hello world {idx}".encode("utf-8")
            group.attrs["phoneme_sequence"] = "HH AH L OW"
            group.attrs["n_time_steps"] = data.shape[0]


def _run_smoke_test() -> None:
    """Comprehensive test of BrainTextDataModule on CPU, MPS, and CUDA."""
    import tempfile

    print("[data] Running tests...")
    
    # Determine available devices
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    
    print(f"[data] Testing on devices: {devices}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        _write_dummy_hdf5(tmp_dir, "train", n_trials=8)
        _write_dummy_hdf5(tmp_dir, "val", n_trials=4)
        
        # Test basic configuration
        data_cfg = BrainTextDataConfig(
            data_root=tmp_dir,
            batch_size=2,
            num_workers=0,  # Avoid multiprocessing in tests
            time_subsample=2,
            max_sequence_length=150,
        )
        
        # Test with character tokenizer
        char_tokenizer = CharTokenizer()
        module = BrainTextDataModule(config=data_cfg, tokenizer=char_tokenizer)
        
        module.prepare_data()
        module.setup()
        
        # Test train dataloader
        train_loader = module.train_dataloader()
        batch = next(iter(train_loader))
        
        assert batch["features"].shape[0] == data_cfg.batch_size
        assert batch["targets"].numel() > 0
        assert batch["feature_lengths"].numel() == data_cfg.batch_size
        assert batch["target_lengths"].numel() == data_cfg.batch_size
        assert len(batch["transcripts"]) == data_cfg.batch_size
        assert all(isinstance(t, str) for t in batch["transcripts"])
        
        # Check time subsampling worked
        original_time = 200
        expected_time = original_time // data_cfg.time_subsample
        max_time = batch["features"].shape[1]
        assert max_time <= expected_time + 1  # Allow for rounding
        
        # Test val dataloader
        val_loader = module.val_dataloader()
        val_batch = next(iter(val_loader))
        assert val_batch["features"].shape[0] <= data_cfg.batch_size
        
        # Test with phoneme tokenizer
        phoneme_tokenizer = PhonemeTokenizer()
        module_with_phonemes = BrainTextDataModule(
            config=data_cfg,
            tokenizer=char_tokenizer,
            phoneme_tokenizer=phoneme_tokenizer,
        )
        module_with_phonemes.setup()
        
        phoneme_batch = next(iter(module_with_phonemes.train_dataloader()))
        assert phoneme_batch["phoneme_targets"] is not None
        assert phoneme_batch["phoneme_lengths"] is not None
        
        # Test on different devices
        for device in devices:
            try:
                batch_device = {
                    "features": batch["features"].to(device),
                    "feature_lengths": batch["feature_lengths"].to(device),
                    "targets": batch["targets"].to(device),
                    "target_lengths": batch["target_lengths"].to(device),
                    "transcripts": batch["transcripts"],
                }
                
                assert batch_device["features"].device.type == device
                assert batch_device["feature_lengths"].device.type == device
                
                print(f"[data] ✓ Batch tensors work on {device}")
            except Exception as e:
                print(f"[data] ✗ Failed on {device}: {e}")
        
        # Test scan_hdf5_trials
        per_split = scan_hdf5_trials(
            data_root=tmp_dir,
            splits=["train", "val"],
            session_limit=None,
            max_trials_per_split=None,
            require_text=True,
        )
        
        assert "train" in per_split
        assert "val" in per_split
        assert len(per_split["train"]) == 8
        assert len(per_split["val"]) == 4
        
        # Test with limits
        per_split_limited = scan_hdf5_trials(
            data_root=tmp_dir,
            splits=["train"],
            session_limit=1,
            max_trials_per_split=3,
            require_text=True,
        )
        assert len(per_split_limited["train"]) <= 3
        
        print("[data] ✓ All tests passed")


if __name__ == "__main__":  # pragma: no cover
    _run_smoke_test()

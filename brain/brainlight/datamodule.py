"""
BrainCLIP Lightning DataModule
==============================

PyTorch Lightning DataModule for managing:
- Train/val/test data loading
- Data preprocessing and augmentation
- Normalization statistics
"""

import lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from pathlib import Path

import sys
sys.path.append('..')

from data.dataset import (
    NEJMBrainTextDataset,
    BrainTextCollator,
)
from data.preprocessing import (
    NeuralSignalAugmentation,
    AugmentationConfig,
)


class BrainCLIPDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for BrainCLIP.
    
    Handles:
    - Dataset creation for train/val/test splits
    - Data augmentation configuration
    - Normalization statistics sharing across splits
    - DataLoader creation with proper collation
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        max_seq_len: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        tokenizer: Optional[Any] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
        use_augmentation: bool = True,
    ):
        """
        Initialize DataModule.
        
        Args:
            data_dir: Path to hdf5_data_final directory
            batch_size: Batch size for all dataloaders
            max_seq_len: Maximum sequence length (time bins)
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            tokenizer: Text tokenizer (will be created if None)
            augmentation_config: Configuration for data augmentation
            use_augmentation: Whether to apply augmentation during training
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tokenizer = tokenizer
        self.use_augmentation = use_augmentation
        
        # Augmentation config
        if augmentation_config is None:
            augmentation_config = AugmentationConfig()
        self.augmentation_config = augmentation_config
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalization_stats = None
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['tokenizer'])
    
    def prepare_data(self):
        """
        Download or prepare data (called on single GPU).
        
        For NEJM data, we assume it's already downloaded.
        This method could be extended to download from Dryad.
        """
        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Please download the NEJM Brain-to-Text dataset from Dryad."
            )
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        
        Args:
            stage: One of 'fit', 'validate', 'test', or None (all)
        """
        # Create augmentation transform
        if self.use_augmentation:
            train_transform = NeuralSignalAugmentation(self.augmentation_config)
        else:
            train_transform = None
        
        # Create training dataset (always needed for normalization stats)
        if stage == 'fit' or stage is None:
            self.train_dataset = NEJMBrainTextDataset(
                data_dir=str(self.data_dir),
                split='train',
                max_seq_len=self.max_seq_len,
                transform=train_transform,
                tokenizer=self.tokenizer,
                normalize=True,
            )
            
            # Store normalization stats for other splits
            self.normalization_stats = self.train_dataset.get_normalization_stats()
        
        # Validation dataset
        if stage == 'fit' or stage == 'validate' or stage is None:
            # Ensure we have normalization stats
            if self.normalization_stats is None and self.train_dataset is not None:
                self.normalization_stats = self.train_dataset.get_normalization_stats()
            
            self.val_dataset = NEJMBrainTextDataset(
                data_dir=str(self.data_dir),
                split='val',
                max_seq_len=self.max_seq_len,
                transform=None,  # No augmentation for validation
                tokenizer=self.tokenizer,
                normalize=True,
                normalization_stats=self.normalization_stats,
            )
        
        # Test dataset
        if stage == 'test' or stage is None:
            if self.normalization_stats is None and self.train_dataset is not None:
                self.normalization_stats = self.train_dataset.get_normalization_stats()
            
            self.test_dataset = NEJMBrainTextDataset(
                data_dir=str(self.data_dir),
                split='test',
                max_seq_len=self.max_seq_len,
                transform=None,
                tokenizer=self.tokenizer,
                normalize=True,
                normalization_stats=self.normalization_stats,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        collator = BrainTextCollator(tokenizer=self.tokenizer)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collator,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        collator = BrainTextCollator(tokenizer=self.tokenizer)
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collator,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        collator = BrainTextCollator(tokenizer=self.tokenizer)
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collator,
        )
    
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for all datasets."""
        self.tokenizer = tokenizer
        
        if self.train_dataset is not None:
            self.train_dataset.tokenizer = tokenizer
        if self.val_dataset is not None:
            self.val_dataset.tokenizer = tokenizer
        if self.test_dataset is not None:
            self.test_dataset.tokenizer = tokenizer


def create_datamodule(
    data_dir: str,
    batch_size: int = 64,
    max_seq_len: int = 256,
    num_workers: int = 4,
    tokenizer: Optional[Any] = None,
    use_augmentation: bool = True,
    # Augmentation settings
    time_mask_prob: float = 0.5,
    channel_dropout_prob: float = 0.2,
    noise_prob: float = 0.5,
    noise_std: float = 0.1,
) -> BrainCLIPDataModule:
    """
    Factory function to create DataModule with common settings.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of workers
        tokenizer: Optional tokenizer
        use_augmentation: Whether to use augmentation
        time_mask_prob: Probability of time masking
        channel_dropout_prob: Probability of channel dropout
        noise_prob: Probability of adding noise
        noise_std: Standard deviation of noise
        
    Returns:
        Configured BrainCLIPDataModule
    """
    aug_config = AugmentationConfig(
        time_mask_enabled=True,
        time_mask_prob=time_mask_prob,
        channel_dropout_enabled=True,
        channel_dropout_prob=channel_dropout_prob,
        noise_enabled=True,
        noise_prob=noise_prob,
        noise_std=noise_std,
    )
    
    return BrainCLIPDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
        tokenizer=tokenizer,
        augmentation_config=aug_config,
        use_augmentation=use_augmentation,
    )

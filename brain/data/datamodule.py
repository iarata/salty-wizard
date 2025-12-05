"""
NEJM Brain-to-Text DataModule
=============================

PyTorch Lightning DataModule specifically designed for:
- Contrastive learning (BrainCLIP)
- Decoder training (text generation)

Supports both retrieval and generation training paradigms.
"""

import lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.append('..')

from data.dataset import NEJMBrainTextDataset
from data.preprocessing import NeuralSignalAugmentation, AugmentationConfig


@dataclass
class DataModuleConfig:
    """Configuration for the data module."""
    data_dir: str = "data/hdf5_data_final"
    batch_size: int = 64
    max_seq_len: int = 512          # Max brain signal sequence length
    max_text_len: int = 128         # Max text token length
    num_workers: int = 4
    pin_memory: bool = True
    use_augmentation: bool = True
    
    # Augmentation settings
    time_mask_prob: float = 0.5
    channel_dropout_prob: float = 0.2
    noise_prob: float = 0.5
    noise_std: float = 0.1


class BrainTextCollator:
    """
    Collator for batching brain-text pairs.
    
    Handles:
    - Padding neural features to same length
    - Tokenizing and padding text
    - Creating attention masks
    - Preparing labels for language modeling
    """
    
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_text_len: int = 128,
        for_generation: bool = False,
    ):
        """
        Args:
            tokenizer: Text tokenizer (HuggingFace)
            max_text_len: Maximum text sequence length
            for_generation: If True, prepare labels for causal LM training
        """
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.for_generation = for_generation
        
        # Ensure tokenizer has pad token
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries from dataset
            
        Returns:
            Batched dictionary with padded tensors
        """
        # Stack neural features (already padded by dataset)
        neural_features = torch.stack([s['neural_features'] for s in batch])
        attention_mask = torch.stack([s['attention_mask'] for s in batch])
        
        # Collect texts
        texts = [s.get('text', '') for s in batch]
        
        # Tokenize text if tokenizer available
        if self.tokenizer is not None and any(texts):
            # Filter empty texts for tokenization
            valid_texts = [t if t else " " for t in texts]  # Replace empty with space
            
            encoding = self.tokenizer(
                valid_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt',
            )
            
            text_tokens = encoding['input_ids']
            text_attention_mask = encoding['attention_mask']
            
            # For generation training, create labels
            if self.for_generation:
                # Labels are same as input_ids, but we'll handle padding in the model
                labels = text_tokens.clone()
                # Set padding tokens to -100 (ignore in loss)
                labels[text_attention_mask == 0] = -100
            else:
                labels = None
        else:
            text_tokens = None
            text_attention_mask = None
            labels = None
        
        result = {
            'neural_features': neural_features,
            'attention_mask': attention_mask,
            'texts': texts,
        }
        
        if text_tokens is not None:
            result['text_tokens'] = text_tokens
            result['text_attention_mask'] = text_attention_mask
        
        if labels is not None:
            result['labels'] = labels
        
        return result


class NEJMBrainTextDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for NEJM Brain-to-Text dataset.
    
    Supports:
    - Contrastive learning (BrainCLIP)
    - Decoder training (text generation)
    - Retrieval evaluation
    
    Key features:
    - Automatic normalization stats from training set
    - Configurable augmentation
    - Proper handling of test set without labels
    """
    
    def __init__(
        self,
        data_dir: str = "data/hdf5_data_final",
        batch_size: int = 64,
        max_seq_len: int = 512,
        max_text_len: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        tokenizer: Optional[Any] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
        use_augmentation: bool = True,
        for_generation: bool = False,
    ):
        """
        Initialize DataModule.
        
        Args:
            data_dir: Path to hdf5_data_final directory
            batch_size: Batch size for all dataloaders
            max_seq_len: Maximum brain signal sequence length (time bins)
            max_text_len: Maximum text token length
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            tokenizer: Text tokenizer (HuggingFace-style)
            augmentation_config: Configuration for data augmentation
            use_augmentation: Whether to apply augmentation during training
            for_generation: If True, prepare data for decoder/generation training
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tokenizer = tokenizer
        self.use_augmentation = use_augmentation
        self.for_generation = for_generation
        
        # Augmentation config
        if augmentation_config is None:
            augmentation_config = AugmentationConfig()
        self.augmentation_config = augmentation_config
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalization_stats = None
        
        # Save hyperparameters (exclude tokenizer as it's not serializable)
        self.save_hyperparameters(ignore=['tokenizer', 'augmentation_config'])
    
    def prepare_data(self):
        """
        Verify data exists (called on single GPU/process).
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Please download the NEJM Brain-to-Text dataset."
            )
        
        # Check for required splits
        session_dirs = list(self.data_dir.glob("t15.*"))
        if not session_dirs:
            raise FileNotFoundError(
                f"No session directories found in {self.data_dir}\n"
                "Expected directories like t15.2023.08.11/"
            )
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        
        Args:
            stage: One of 'fit', 'validate', 'test', 'predict', or None (all)
        """
        # Create augmentation transform for training
        if self.use_augmentation:
            train_transform = NeuralSignalAugmentation(self.augmentation_config)
        else:
            train_transform = None
        
        # Training dataset (always create for normalization stats)
        if stage in ('fit', None) or self.normalization_stats is None:
            self.train_dataset = NEJMBrainTextDataset(
                data_dir=str(self.data_dir),
                split='train',
                max_seq_len=self.max_seq_len,
                transform=train_transform,
                tokenizer=self.tokenizer,
                normalize=True,
            )
            self.normalization_stats = self.train_dataset.get_normalization_stats()
        
        # Validation dataset
        if stage in ('fit', 'validate', None):
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
        if stage in ('test', 'predict', None):
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
        collator = BrainTextCollator(
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            for_generation=self.for_generation,
        )
        
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
        collator = BrainTextCollator(
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            for_generation=self.for_generation,
        )
        
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
        collator = BrainTextCollator(
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            for_generation=self.for_generation,
        )
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collator,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader (same as test)."""
        return self.test_dataloader()
    
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for all datasets and collators."""
        self.tokenizer = tokenizer
        
        # Ensure pad token exists
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Update datasets
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            if dataset is not None:
                dataset.tokenizer = tokenizer
    
    def get_normalization_stats(self) -> Tuple:
        """Return normalization statistics (mean, std)."""
        return self.normalization_stats
    
    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        return len(self.train_dataset) if self.train_dataset else 0
    
    @property
    def num_val_samples(self) -> int:
        """Number of validation samples."""
        return len(self.val_dataset) if self.val_dataset else 0
    
    @property
    def num_test_samples(self) -> int:
        """Number of test samples."""
        return len(self.test_dataset) if self.test_dataset else 0


def create_datamodule(
    data_dir: str,
    batch_size: int = 64,
    max_seq_len: int = 512,
    max_text_len: int = 128,
    num_workers: int = 4,
    tokenizer: Optional[Any] = None,
    use_augmentation: bool = True,
    for_generation: bool = False,
    # Augmentation settings
    time_mask_prob: float = 0.5,
    channel_dropout_prob: float = 0.2,
    noise_prob: float = 0.5,
    noise_std: float = 0.1,
) -> NEJMBrainTextDataModule:
    """
    Factory function to create DataModule with common settings.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        max_seq_len: Maximum brain signal sequence length
        max_text_len: Maximum text token length
        num_workers: Number of workers
        tokenizer: Optional tokenizer
        use_augmentation: Whether to use augmentation
        for_generation: If True, prepare data for decoder training
        time_mask_prob: Probability of time masking
        channel_dropout_prob: Probability of channel dropout
        noise_prob: Probability of adding noise
        noise_std: Standard deviation of noise
        
    Returns:
        Configured NEJMBrainTextDataModule
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
    
    return NEJMBrainTextDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_text_len=max_text_len,
        num_workers=num_workers,
        tokenizer=tokenizer,
        augmentation_config=aug_config,
        use_augmentation=use_augmentation,
        for_generation=for_generation,
    )


# Alias for backward compatibility
BrainCLIPDataModule = NEJMBrainTextDataModule

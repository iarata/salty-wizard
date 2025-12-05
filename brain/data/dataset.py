"""
========================

Dataset classes for loading and preprocessing the NEJM Brain-to-Text dataset.
The data is stored in HDF5 files with the following structure per trial:
- input_features: (T, 512) neural features (combined threshold crossings + spike band power)
- seq_class_ids: phoneme sequence labels
- transcription: character-level transcription
- Attributes: block_num, trial_num, sentence_label, n_time_steps, seq_len, session
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NEJMBrainTextDataset(Dataset):
    """
    Dataset for NEJM Brain-to-Text neural recordings.
    
    Loads neural features (threshold crossings + spike band power) and 
    corresponding text labels from HDF5 files.
    
    Args:
        data_dir: Path to hdf5_data_final directory
        split: One of 'train', 'val', 'test'
        max_seq_len: Maximum sequence length (time bins)
        transform: Optional transform function for augmentation
        tokenizer: Text tokenizer for encoding sentences
        normalize: Whether to apply z-score normalization
        normalization_stats: Pre-computed (mean, std) for normalization
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = 256,
        transform: Optional[callable] = None,
        tokenizer: Optional[Any] = None,
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.tokenizer = tokenizer
        self.normalize = normalize
        
        # Load all trials from HDF5 files
        self.trials = self._load_trials()
        
        # Compute or use provided normalization statistics
        if normalize:
            if normalization_stats is not None:
                self.mean, self.std = normalization_stats
            else:
                self.mean, self.std = self._compute_normalization_stats()
        else:
            self.mean, self.std = None, None
            
        logger.info(f"Loaded {len(self.trials)} {split} trials")
    
    def _load_trials(self) -> List[Dict]:
        """
        Load all trials from HDF5 files for the specified split.
        
        Returns:
            List of trial dictionaries with file path and trial info
        """
        trials = []
        
        # Find all session directories
        session_dirs = sorted(glob.glob(str(self.data_dir / "t15.*")))
        
        for session_dir in session_dirs:
            # Look for split-specific file
            split_file = os.path.join(session_dir, f"data_{self.split}.hdf5")
            
            if os.path.exists(split_file):
                # Load trial indices from this file
                with h5py.File(split_file, 'r') as f:
                    # Each group in the file is a trial
                    for trial_key in f.keys():
                        trials.append({
                            'file_path': split_file,
                            'trial_key': trial_key,
                            'session': os.path.basename(session_dir),
                        })
        
        return trials
    
    def _compute_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std across all training data for normalization.
        
        Returns:
            Tuple of (mean, std) arrays of shape (512,)
        """
        logger.info("Computing normalization statistics...")
        
        all_features = []
        
        # Sample subset for efficiency
        sample_indices = np.random.choice(
            len(self.trials), 
            min(1000, len(self.trials)), 
            replace=False
        )
        
        for idx in sample_indices:
            trial = self.trials[idx]
            with h5py.File(trial['file_path'], 'r') as f:
                grp = f[trial['trial_key']]
                # Neural features are already combined in input_features (T, 512)
                features = np.array(grp['input_features'])
                all_features.append(features)
        
        all_features = np.concatenate(all_features, axis=0)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0) + 1e-8  # Avoid division by zero
        
        return mean, std
    
    def _load_trial_data(self, trial: Dict) -> Dict[str, Any]:
        """
        Load a single trial's data from HDF5.
        
        Args:
            trial: Trial dictionary with file path and key
            
        Returns:
            Dictionary with neural features and text
        """
        with h5py.File(trial['file_path'], 'r') as f:
            grp = f[trial['trial_key']]
            
            # Load neural features - already combined (T, 512)
            neural_features = np.array(grp['input_features'], dtype=np.float32)
            
            # Load text from attribute 'sentence_label'
            sentence_text = grp.attrs.get('sentence_label', '')
            if isinstance(sentence_text, bytes):
                sentence_text = sentence_text.decode('utf-8')
            
            # Load phonemes if available (seq_class_ids)
            phonemes = None
            if 'seq_class_ids' in grp:
                phonemes = np.array(grp['seq_class_ids'])
            
            # Metadata from attributes
            block_idx = int(grp.attrs.get('block_num', -1))
            trial_idx = int(grp.attrs.get('trial_num', -1))
        
        return {
            'neural_features': neural_features,
            'text': sentence_text,
            'phonemes': phonemes,
            'block_idx': block_idx,
            'trial_idx': trial_idx,
            'session': trial['session'],
            'seq_len': neural_features.shape[0],
        }
    
    def _normalize_features(
        self, 
        features: np.ndarray
    ) -> np.ndarray:
        """Apply z-score normalization."""
        if self.mean is not None and self.std is not None:
            return (features - self.mean) / self.std
        return features
    
    def _pad_or_truncate(
        self, 
        features: np.ndarray,
        target_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or truncate features to target length.
        
        Args:
            features: (T, F) array
            target_len: Target sequence length
            
        Returns:
            Tuple of (padded_features, attention_mask)
        """
        T, F = features.shape
        
        if T >= target_len:
            # Truncate
            features = features[:target_len]
            mask = np.ones(target_len, dtype=np.float32)
        else:
            # Pad with zeros
            pad_len = target_len - T
            features = np.pad(
                features, 
                ((0, pad_len), (0, 0)), 
                mode='constant', 
                constant_values=0
            )
            mask = np.concatenate([
                np.ones(T, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32)
            ])
        
        return features, mask
    
    def __len__(self) -> int:
        return len(self.trials)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            Dictionary with:
                - neural_features: (max_seq_len, 512) tensor
                - attention_mask: (max_seq_len,) tensor
                - text: Original text string
                - text_tokens: Tokenized text (if tokenizer provided)
                - text_attention_mask: Text attention mask
        """
        trial = self.trials[idx]
        data = self._load_trial_data(trial)
        
        neural_features = data['neural_features']
        
        # Normalize
        if self.normalize:
            neural_features = self._normalize_features(neural_features)
        
        # Apply augmentation (training only)
        if self.transform is not None:
            neural_features = self.transform(neural_features)
        
        # Pad or truncate
        neural_features, attention_mask = self._pad_or_truncate(
            neural_features, 
            self.max_seq_len
        )
        
        # Convert to tensors
        output = {
            'neural_features': torch.from_numpy(neural_features),
            'attention_mask': torch.from_numpy(attention_mask),
            'text': data['text'],
            'seq_len': data['seq_len'],
        }
        
        # Tokenize text if tokenizer provided
        if self.tokenizer is not None:
            text_encoding = self.tokenizer(
                data['text'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            output['text_tokens'] = text_encoding['input_ids'].squeeze(0)
            output['text_attention_mask'] = text_encoding['attention_mask'].squeeze(0)
        
        return output
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalization statistics for use with other splits."""
        return self.mean, self.std


class BrainTextCollator:
    """
    Custom collator for batching brain-text pairs.
    
    Handles variable-length sequences and text tokenization.
    """
    
    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            batch: List of dictionaries from dataset
            
        Returns:
            Batched dictionary
        """
        # Stack neural features
        neural_features = torch.stack([b['neural_features'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        
        # Collect texts
        texts = [b['text'] for b in batch]
        
        output = {
            'neural_features': neural_features,
            'attention_mask': attention_mask,
            'texts': texts,
        }
        
        # Stack tokenized text if available
        if 'text_tokens' in batch[0]:
            output['text_tokens'] = torch.stack([b['text_tokens'] for b in batch])
            output['text_attention_mask'] = torch.stack([b['text_attention_mask'] for b in batch])
        
        return output


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    max_seq_len: int = 256,
    tokenizer: Optional[Any] = None,
    train_transform: Optional[callable] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        tokenizer: Text tokenizer
        train_transform: Augmentation for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create training dataset first to get normalization stats
    train_dataset = NEJMBrainTextDataset(
        data_dir=data_dir,
        split='train',
        max_seq_len=max_seq_len,
        transform=train_transform,
        tokenizer=tokenizer,
        normalize=True,
    )
    
    # Get normalization stats from training set
    norm_stats = train_dataset.get_normalization_stats()
    
    # Create validation dataset with same normalization
    val_dataset = NEJMBrainTextDataset(
        data_dir=data_dir,
        split='val',
        max_seq_len=max_seq_len,
        transform=None,  # No augmentation for validation
        tokenizer=tokenizer,
        normalize=True,
        normalization_stats=norm_stats,
    )
    
    # Create test dataset
    test_dataset = NEJMBrainTextDataset(
        data_dir=data_dir,
        split='test',
        max_seq_len=max_seq_len,
        transform=None,
        tokenizer=tokenizer,
        normalize=True,
        normalization_stats=norm_stats,
    )
    
    # Create collator
    collator = BrainTextCollator(tokenizer=tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    
    return train_loader, val_loader, test_loader

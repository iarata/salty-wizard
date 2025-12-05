"""
BrainCLIP Preprocessing and Augmentation Module
================================================

Data augmentation strategies for neural signal data:
- Time masking (SpecAugment-style)
- Channel/electrode dropout
- Gaussian noise injection
- Time shifting
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Time masking (SpecAugment-style) - BALANCED
    time_mask_enabled: bool = True
    time_mask_prob: float = 0.5           # Moderate
    time_mask_num_masks: int = 2          
    time_mask_ratio_min: float = 0.05
    time_mask_ratio_max: float = 0.2     
    
    # Channel dropout - BALANCED
    channel_dropout_enabled: bool = True
    channel_dropout_prob: float = 0.2     
    channel_drop_arrays: bool = True  # Drop entire arrays vs individual channels
    
    # Gaussian noise - BALANCED
    noise_enabled: bool = True
    noise_prob: float = 0.5               
    noise_std: float = 0.1               
    
    # Time shifting
    time_shift_enabled: bool = True
    time_shift_prob: float = 0.3          
    time_shift_max: int = 5              
    
    # Time warping (slight temporal distortion)
    time_warp_enabled: bool = True
    time_warp_prob: float = 0.2
    time_warp_factor: float = 0.1         # Max 10% stretch/compress
    
    # Mixup-style augmentation probability
    mixup_enabled: bool = False           # Disabled by default
    mixup_prob: float = 0.3
    mixup_alpha: float = 0.2


class NeuralSignalAugmentation:
    """
    Augmentation transforms for neural signal data.
    
    Applies various augmentations to improve model robustness:
    1. Time masking: Masks random time spans (similar to SpecAugment)
    2. Channel dropout: Drops electrode arrays or channels
    3. Gaussian noise: Adds random noise
    4. Time shifting: Shifts the signal temporally
    5. Time warping: Slight temporal stretch/compress
    
    All augmentations preserve the overall structure while
    creating diversity in training data.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to neural features.
        
        Args:
            features: (T, 512) array of neural features
            
        Returns:
            Augmented features of same shape
        """
        features = features.copy()  # Don't modify original
        
        # Time warping (do first as it changes time dimension)
        if getattr(self.config, 'time_warp_enabled', False) and random.random() < self.config.time_warp_prob:
            features = self._time_warp(features)
        
        # Time masking
        if self.config.time_mask_enabled and random.random() < self.config.time_mask_prob:
            features = self._time_mask(features)
        
        # Channel dropout
        if self.config.channel_dropout_enabled and random.random() < self.config.channel_dropout_prob:
            features = self._channel_dropout(features)
        
        # Gaussian noise
        if self.config.noise_enabled and random.random() < self.config.noise_prob:
            features = self._add_noise(features)
        
        # Time shifting
        if self.config.time_shift_enabled and random.random() < self.config.time_shift_prob:
            features = self._time_shift(features)
        
        return features
    
    def _time_warp(self, features: np.ndarray) -> np.ndarray:
        """
        Apply time warping (stretch or compress time axis).
        
        Args:
            features: (T, F) array
            
        Returns:
            Time-warped features (same shape as input)
        """
        T, F = features.shape
        
        # Random warp factor
        warp_factor = 1.0 + random.uniform(-self.config.time_warp_factor, self.config.time_warp_factor)
        
        # Create new time indices
        new_T = int(T * warp_factor)
        if new_T < 2:
            return features
        
        # Interpolate to new length
        old_indices = np.arange(T)
        new_indices = np.linspace(0, T - 1, new_T)
        
        # Interpolate each feature channel
        warped = np.zeros((new_T, F), dtype=features.dtype)
        for f in range(F):
            warped[:, f] = np.interp(new_indices, old_indices, features[:, f])
        
        # Resize back to original length (crop or pad)
        if new_T > T:
            # Crop from center
            start = (new_T - T) // 2
            warped = warped[start:start + T]
        elif new_T < T:
            # Pad symmetrically
            pad_before = (T - new_T) // 2
            pad_after = T - new_T - pad_before
            warped = np.pad(warped, ((pad_before, pad_after), (0, 0)), mode='edge')
        
        return warped
    
    def _time_mask(self, features: np.ndarray) -> np.ndarray:
        """
        Apply time masking (mask random time spans with zeros).
        
        Similar to SpecAugment for speech recognition.
        
        Args:
            features: (T, F) array
            
        Returns:
            Features with masked time spans
        """
        T, F = features.shape
        
        for _ in range(self.config.time_mask_num_masks):
            # Random mask length
            mask_ratio = random.uniform(
                self.config.time_mask_ratio_min,
                self.config.time_mask_ratio_max
            )
            mask_len = int(T * mask_ratio)
            
            if mask_len > 0:
                # Random start position
                start = random.randint(0, max(0, T - mask_len))
                end = min(start + mask_len, T)
                
                # Apply mask
                features[start:end, :] = 0.0
        
        return features
    
    def _channel_dropout(self, features: np.ndarray) -> np.ndarray:
        """
        Drop electrode arrays or individual channels.
        
        The 512 features are organized as:
        - 0-255: Threshold crossings (4 arrays × 64 channels)
        - 256-511: Spike band power (4 arrays × 64 channels)
        
        For each array (e.g., ventral 6v):
        - TX: 0-63, SPow: 256-319
        
        Args:
            features: (T, 512) array
            
        Returns:
            Features with dropped channels
        """
        T, F = features.shape
        
        if self.config.channel_drop_arrays:
            # Drop entire electrode arrays
            # Array mapping (both TX and SPow for each array):
            # Array 0 (v6v): TX 0-63, SPow 256-319
            # Array 1 (area4): TX 64-127, SPow 320-383
            # Array 2 (55b): TX 128-191, SPow 384-447
            # Array 3 (d6v): TX 192-255, SPow 448-511
            
            array_to_drop = random.randint(0, 3)
            
            # Zero out both TX and SPow for this array
            tx_start = array_to_drop * 64
            tx_end = tx_start + 64
            spow_start = 256 + array_to_drop * 64
            spow_end = spow_start + 64
            
            features[:, tx_start:tx_end] = 0.0
            features[:, spow_start:spow_end] = 0.0
        else:
            # Drop random individual channels
            num_to_drop = random.randint(1, 64)
            channels_to_drop = random.sample(range(F), num_to_drop)
            features[:, channels_to_drop] = 0.0
        
        return features
    
    def _add_noise(self, features: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to features.
        
        Noise level is relative to feature standard deviation.
        
        Args:
            features: (T, F) array
            
        Returns:
            Features with added noise
        """
        noise = np.random.randn(*features.shape).astype(np.float32)
        noise *= self.config.noise_std
        
        return features + noise
    
    def _time_shift(self, features: np.ndarray) -> np.ndarray:
        """
        Shift features temporally (circular shift).
        
        Args:
            features: (T, F) array
            
        Returns:
            Time-shifted features
        """
        shift = random.randint(-self.config.time_shift_max, self.config.time_shift_max)
        
        if shift != 0:
            features = np.roll(features, shift, axis=0)
            
            # Zero out the wrapped region
            if shift > 0:
                features[:shift, :] = 0.0
            else:
                features[shift:, :] = 0.0
        
        return features


class FeatureNormalizer:
    """
    Feature normalization utilities.
    
    Supports:
    - Z-score normalization (per-feature mean/std)
    - Per-session normalization
    - Robust normalization (median/IQR)
    """
    
    def __init__(
        self,
        method: str = "zscore",
        per_session: bool = False,
    ):
        self.method = method
        self.per_session = per_session
        self.stats = {}
    
    def fit(
        self, 
        features: np.ndarray,
        session_ids: Optional[List[str]] = None
    ):
        """
        Compute normalization statistics from data.
        
        Args:
            features: (N, T, F) or list of (T, F) arrays
            session_ids: Optional session identifiers for per-session norm
        """
        if self.per_session and session_ids is not None:
            # Compute stats per session
            unique_sessions = set(session_ids)
            
            for session in unique_sessions:
                session_mask = [s == session for s in session_ids]
                session_features = [f for f, m in zip(features, session_mask) if m]
                session_data = np.concatenate(session_features, axis=0)
                
                self.stats[session] = self._compute_stats(session_data)
        else:
            # Global statistics
            if isinstance(features, list):
                all_data = np.concatenate(features, axis=0)
            else:
                all_data = features.reshape(-1, features.shape[-1])
            
            self.stats['global'] = self._compute_stats(all_data)
    
    def _compute_stats(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistics for a dataset."""
        if self.method == "zscore":
            return {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0) + 1e-8,
            }
        elif self.method == "robust":
            return {
                'median': np.median(data, axis=0),
                'iqr': np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0) + 1e-8,
            }
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def transform(
        self, 
        features: np.ndarray,
        session_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply normalization to features.
        
        Args:
            features: (T, F) array
            session_id: Optional session identifier
            
        Returns:
            Normalized features
        """
        if self.per_session and session_id is not None and session_id in self.stats:
            stats = self.stats[session_id]
        else:
            stats = self.stats.get('global', self.stats.get(list(self.stats.keys())[0]))
        
        if self.method == "zscore":
            return (features - stats['mean']) / stats['std']
        elif self.method == "robust":
            return (features - stats['median']) / (stats['iqr'] * 0.7413)  # Scale to ~std
        else:
            return features


def reorganize_by_array(
    features: np.ndarray,
    num_arrays: int = 4,
    features_per_array: int = 128
) -> np.ndarray:
    """
    Reorganize features by electrode array for spatial attention.
    
    Input features (512) are organized as:
    [TX_v6v, TX_area4, TX_55b, TX_d6v, SPow_v6v, SPow_area4, SPow_55b, SPow_d6v]
    
    Output reorganizes to group TX and SPow for each array:
    Array 0: [TX_v6v(64), SPow_v6v(64)]
    Array 1: [TX_area4(64), SPow_area4(64)]
    etc.
    
    Args:
        features: (T, 512) or (B, T, 512) array
        num_arrays: Number of electrode arrays (4)
        features_per_array: Features per array after reorganization (128)
        
    Returns:
        (T, num_arrays, features_per_array) or (B, T, num_arrays, features_per_array)
    """
    has_batch = features.ndim == 3
    
    if not has_batch:
        features = features[np.newaxis, ...]  # Add batch dim
    
    B, T, F = features.shape
    assert F == 512, f"Expected 512 features, got {F}"
    
    # Split into TX and SPow
    tx = features[..., :256]      # (B, T, 256)
    spow = features[..., 256:]    # (B, T, 256)
    
    # Reshape and interleave
    tx_arrays = tx.reshape(B, T, num_arrays, 64)      # (B, T, 4, 64)
    spow_arrays = spow.reshape(B, T, num_arrays, 64)  # (B, T, 4, 64)
    
    # Concatenate TX and SPow for each array
    # Result: (B, T, 4, 128)
    result = np.concatenate([tx_arrays, spow_arrays], axis=-1)
    
    if not has_batch:
        result = result.squeeze(0)
    
    return result


def create_patch_embedding(
    features: np.ndarray,
    patch_size: int = 5,
    flatten: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Create patches from temporal features.
    
    Args:
        features: (T, F) array
        patch_size: Number of time bins per patch
        flatten: Whether to flatten patches
        
    Returns:
        Tuple of (patches, num_patches)
        - If flatten: (num_patches, patch_size * F)
        - If not flatten: (num_patches, patch_size, F)
    """
    T, F = features.shape
    
    # Truncate to exact multiple of patch_size
    num_patches = T // patch_size
    features = features[:num_patches * patch_size]
    
    # Reshape into patches
    patches = features.reshape(num_patches, patch_size, F)
    
    if flatten:
        patches = patches.reshape(num_patches, -1)
    
    return patches, num_patches
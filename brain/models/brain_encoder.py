"""
BrainCLIP Brain Encoder Module
==============================

Spatiotemporal transformer encoder for neural signals, inspired by NuCLR.

Architecture:
1. Patch embedding: Convert temporal patches to embeddings
2. Temporal transformer: Process each electrode array's temporal dynamics
3. Spatiotemporal transformer: Exchange information across arrays and time
4. Pooling and projection: Produce fixed-size brain embedding

Key features:
- Permutation equivariant across electrode arrays
- Rotary position embeddings for temporal structure
- Separate spatial and temporal attention mechanisms
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BrainEncoderConfig:
    """Configuration for brain encoder."""
    
    # Input dimensions
    num_features: int = 512           # Total neural features
    num_arrays: int = 4               # Number of electrode arrays
    features_per_array: int = 128     # Features per array (64 TX + 64 SPow)
    
    # Patching
    patch_size: int = 5               # Time bins per patch (100ms at 20ms bins)
    
    # Transformer architecture
    hidden_dim: int = 256             # Embedding dimension
    num_temporal_layers: int = 4      # Pure temporal transformer layers
    num_spatiotemporal_layers: int = 4  # Alternating spatial-temporal layers
    num_heads: int = 8
    ffn_dim: int = 1024               # Feed-forward dimension
    dropout: float = 0.2              # Balanced dropout
    attention_dropout: float = 0.1    # Moderate attention dropout
    
    # Position encoding
    use_rope: bool = True
    max_positions: int = 512
    
    # Output
    pool_type: str = "mean"           # "mean", "cls", or "attention"
    
    # Regularization
    layer_drop_rate: float = 0.05     # Moderate stochastic depth


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes relative position information by rotating query and key vectors
    based on their absolute positions. This allows the model to learn
    relative position patterns naturally.
    
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_positions: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.base = base
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_positions)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cosine and sine values for positions."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: (..., seq_len, dim) tensor
            seq_len: Sequence length (optional, inferred from x)
            
        Returns:
            Tensor with same shape, with RoPE applied
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Extend cache if needed
        if seq_len > self.max_positions:
            self._set_cos_sin_cache(seq_len)
        
        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary transformation."""
        # Split into rotation pairs
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_rot = x_rot.flatten(-2)  # Interleave
        
        # Apply rotation
        return x * cos + x_rot * sin


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional RoPE.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_rope: Whether to use rotary position embeddings
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = False,
        max_positions: int = 512,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_positions)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, dim) input tensor
            mask: Optional (batch, seq_len) attention mask
            
        Returns:
            (batch, seq_len, dim) output tensor
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE to Q and K
        # q, k shapes: (B, heads, N, head_dim) - RoPE operates on last two dims
        if self.use_rope:
            q = self.rope(q, N)
            k = self.rope(k, N)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N)
            mask = mask[:, None, None, :]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with stochastic depth.
    
    # Structure:
    # x -> LayerNorm -> MultiHeadAttention -> + -> LayerNorm -> FFN -> +
    
    #                  residual                          residual
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = False,
        max_positions: int = 512,
        drop_path_rate: float = 0.0,  # Stochastic depth
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads, dropout, attention_dropout, use_rope, max_positions
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Stochastic depth: randomly skip this block during training
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1).item() < self.drop_path_rate:
                return x  # Skip this block entirely
        
        # Self-attention with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # FFN with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TemporalTransformer(nn.Module):
    """
    Stack of transformer blocks for temporal processing.
    
    Processes each electrode array's temporal dynamics independently.
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        max_positions: int = 512,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        # Stochastic depth: linearly increase drop rate across layers
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, num_heads, ffn_dim, dropout, attention_dropout,
                use_rope, max_positions, drop_path_rate=drop_rates[i]
            )
            for i in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) or (batch, arrays, seq_len, dim)
            mask: Optional attention mask
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SpatialTransformerBlock(nn.Module):
    """
    Spatial attention block.
    
    At each time point, allows electrodes arrays to exchange information.
    No positional encoding (permutation equivariant across arrays).
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, attention_dropout, use_rope=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * time, arrays, dim) - attention over arrays at each time
        """
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Alternating spatial and temporal transformer layers.
    
    Architecture:
    For each layer:
        1. Spatial attention: Exchange info across arrays at each time point
        2. Temporal attention: Exchange info across time for each array
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        max_positions: int = 512,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        
        # Stochastic depth: linearly increase drop rate across layers
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        for i in range(num_layers):
            self.spatial_layers.append(
                SpatialTransformerBlock(dim, num_heads, ffn_dim, dropout, attention_dropout)
            )
            self.temporal_layers.append(
                TransformerBlock(
                    dim, num_heads, ffn_dim, dropout, attention_dropout,
                    use_rope, max_positions, drop_path_rate=drop_rates[i]
                )
            )
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, arrays, time, dim)
            mask: (batch, time) attention mask
            
        Returns:
            (batch, arrays, time, dim)
        """
        B, A, T, D = x.shape
        
        for spatial, temporal in zip(self.spatial_layers, self.temporal_layers):
            # Spatial attention: (B, A, T, D) -> (B*T, A, D)
            x_spatial = x.permute(0, 2, 1, 3).reshape(B * T, A, D)
            x_spatial = spatial(x_spatial)
            x = x_spatial.reshape(B, T, A, D).permute(0, 2, 1, 3)
            
            # Temporal attention: (B, A, T, D) -> (B*A, T, D)
            x_temporal = x.reshape(B * A, T, D)
            
            # Expand mask for all arrays
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).expand(-1, A, -1).reshape(B * A, T)
            else:
                mask_expanded = None
            
            x_temporal = temporal(x_temporal, mask_expanded)
            x = x_temporal.reshape(B, A, T, D)
        
        return x


class BrainEncoder(nn.Module):
    """
    Complete brain encoder for neural signals.
    
    Pipeline:
    1. Reorganize input by electrode array
    2. Create temporal patches and embed
    3. Temporal transformer (per-array)
    4. Spatiotemporal transformer (cross-array + temporal)
    5. Pool and project to embedding
    
    Input: (batch, time, 512) neural features
    Output: (batch, embedding_dim) brain embedding
    """
    
    def __init__(self, config: BrainEncoderConfig):
        super().__init__()
        self.config = config
        
        # Get attention dropout, defaulting to regular dropout if not specified
        attention_dropout = getattr(config, 'attention_dropout', config.dropout)
        drop_path_rate = getattr(config, 'layer_drop_rate', 0.0)
        
        # Patch embedding with dropout
        # Input: (batch, time, arrays, features_per_array)
        # After patching: (batch, arrays, num_patches, patch_size * features_per_array)
        patch_dim = config.patch_size * config.features_per_array
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        
        # Temporal transformer (processes each array independently)
        self.temporal_transformer = TemporalTransformer(
            dim=config.hidden_dim,
            num_layers=config.num_temporal_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attention_dropout=attention_dropout,
            use_rope=config.use_rope,
            max_positions=config.max_positions,
            drop_path_rate=drop_path_rate,
        )
        
        # Spatiotemporal transformer
        self.spatiotemporal_transformer = SpatioTemporalTransformer(
            dim=config.hidden_dim,
            num_layers=config.num_spatiotemporal_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attention_dropout=attention_dropout,
            use_rope=config.use_rope,
            max_positions=config.max_positions,
            drop_path_rate=drop_path_rate,
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # CLS token for optional cls pooling
        if config.pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        
        # Output dimension: hidden_dim * num_arrays (after concatenating array embeddings)
        self.output_dim = config.hidden_dim * config.num_arrays
    
    def _reorganize_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganize features by electrode array.
        
        Input: (B, T, 512) where 512 = [TX_0..3, SPow_0..3]
        Output: (B, T, 4, 128) where 128 = [TX_i, SPow_i] for each array
        """
        B, T, F = x.shape
        
        # Split threshold crossings and spike band power
        tx = x[..., :256]       # (B, T, 256)
        spow = x[..., 256:]     # (B, T, 256)
        
        # Reshape to separate arrays
        tx = tx.reshape(B, T, 4, 64)       # (B, T, 4, 64)
        spow = spow.reshape(B, T, 4, 64)   # (B, T, 4, 64)
        
        # Concatenate TX and SPow for each array
        # Result: (B, T, 4, 128)
        return torch.cat([tx, spow], dim=-1)
    
    def _create_patches(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Create temporal patches.
        
        Input: (B, T, A, F) where A=arrays, F=features per array
        Output: (B, A, P, patch_size * F) where P=num_patches
        """
        B, T, A, F = x.shape
        P = T // self.config.patch_size  # Number of patches
        
        # Truncate to exact multiple
        x = x[:, :P * self.config.patch_size, :, :]
        
        # Reshape into patches: (B, P, patch_size, A, F)
        x = x.reshape(B, P, self.config.patch_size, A, F)
        
        # Reorder to: (B, A, P, patch_size, F)
        x = x.permute(0, 3, 1, 2, 4)
        
        # Flatten patch: (B, A, P, patch_size * F)
        x = x.reshape(B, A, P, -1)
        
        return x, P
    
    def _create_patch_mask(
        self, 
        mask: torch.Tensor, 
        num_patches: int
    ) -> torch.Tensor:
        """
        Create attention mask for patches.
        
        A patch is valid if ANY time bin in the patch is valid.
        
        Input: (B, T) mask
        Output: (B, P) patch mask
        """
        B, T = mask.shape
        
        # Truncate to patch boundary
        T_truncated = num_patches * self.config.patch_size
        mask = mask[:, :T_truncated]
        
        # Reshape and reduce: (B, P, patch_size) -> (B, P)
        mask = mask.reshape(B, num_patches, self.config.patch_size)
        patch_mask = mask.any(dim=-1).float()
        
        return patch_mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, time, 512) neural features
            mask: (batch, time) attention mask (1=valid, 0=padded)
            
        Returns:
            (batch, hidden_dim * num_arrays) brain embedding
        """
        # 1. Reorganize by electrode array
        x = self._reorganize_features(x)  # (B, T, 4, 128)
        
        # 2. Create patches
        x, num_patches = self._create_patches(x)  # (B, 4, P, patch_dim)
        
        # Create patch-level mask
        if mask is not None:
            patch_mask = self._create_patch_mask(mask, num_patches)
        else:
            patch_mask = None
        
        # 3. Embed patches
        x = self.patch_embed(x)  # (B, 4, P, hidden_dim)
        
        # 4. Temporal transformer (process each array independently)
        B, A, P, D = x.shape
        x = x.reshape(B * A, P, D)  # (B*4, P, D)
        
        if patch_mask is not None:
            temp_mask = patch_mask.unsqueeze(1).expand(-1, A, -1).reshape(B * A, P)
        else:
            temp_mask = None
        
        x = self.temporal_transformer(x, temp_mask)
        x = x.reshape(B, A, P, D)  # (B, 4, P, D)
        
        # 5. Spatiotemporal transformer
        x = self.spatiotemporal_transformer(x, patch_mask)  # (B, 4, P, D)
        
        # 6. Final layer norm
        x = self.final_norm(x)
        
        # 7. Pooling
        if self.config.pool_type == "mean":
            # Mean over time, then concatenate arrays
            if patch_mask is not None:
                # Masked mean
                mask_expanded = patch_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, P, 1)
                x = (x * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2).clamp(min=1)
            else:
                x = x.mean(dim=2)  # (B, 4, D)
            
            # Concatenate array representations
            x = x.reshape(B, -1)  # (B, 4*D)
        
        elif self.config.pool_type == "cls":
            # Use CLS token from first position
            x = x[:, :, 0, :]  # (B, 4, D)
            x = x.reshape(B, -1)  # (B, 4*D)
        
        else:
            raise ValueError(f"Unknown pool type: {self.config.pool_type}")
        
        return x


# Test the model
if __name__ == "__main__":
    config = BrainEncoderConfig()
    model = BrainEncoder(config)
    
    # Test input
    batch_size = 4
    seq_len = 200  # Time bins
    x = torch.randn(batch_size, seq_len, 512)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 180:] = 0  # Mask last 20 bins
    
    # Forward pass
    output = model(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output dim: {model.output_dim}")
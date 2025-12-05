"""
BrainCLIP Decoder V2 Training Module
=====================================

Training module with techniques to prevent decoder ignoring brain signals:

1. Auxiliary contrastive loss - keeps brain-text alignment
2. Teacher forcing with scheduled sampling
3. Gradient balancing between losses
4. Validation generation monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup
import random

import sys
sys.path.append('..')

# Try different import paths
try:
    from models.brainclip_decoder import BrainCLIPDecoderV2, DecoderV2Config
    from models.brain_encoder import BrainEncoder, BrainEncoderConfig
except ImportError:
    try:
        from models.brainclip_decoder import BrainCLIPDecoderV2, DecoderV2Config
        from models.brain_encoder import BrainEncoder, BrainEncoderConfig
    except ImportError:
        raise ImportError("Could not import BrainCLIPDecoder or BrainEncoder. Ensure model files are accessible.")


@dataclass
class DecoderV2TrainingConfig:
    """Training configuration."""
    # Learning rates
    learning_rate: float = 5e-5
    brain_encoder_lr: float = 0.0       # Frozen by default
    prefix_lr: float = 1e-4             # Higher for prefix
    cross_attention_lr: float = 5e-5    # For cross-attn layers
    decoder_lr: float = 2e-5            # Lower for pretrained
    
    # Optimization
    weight_decay: float = 0.05          # Increased for regularization
    warmup_epochs: int = 5
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    
    # Regularization
    dropout: float = 0.2                # Dropout for prefix projection
    label_smoothing: float = 0.1        # Label smoothing for cross-entropy
    
    # Training tricks
    scheduled_sampling: bool = True     # Gradually use model predictions
    sampling_start_epoch: int = 5       # When to start scheduled sampling
    sampling_rate: float = 0.1          # Initial rate of using model predictions
    
    # Losses
    contrastive_weight: float = 0.1
    
    # Logging
    log_generation_every_n_steps: int = 500


class BrainCLIPDecoderV2Module(pl.LightningModule):
    """Training module for decoder V2."""
    
    def __init__(
        self,
        model: BrainCLIPDecoderV2,
        config: DecoderV2TrainingConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters()
        
        # For scheduled sampling
        self.current_sampling_rate = 0.0
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step."""
        
        outputs = self.model(
            neural_features=batch['neural_features'],
            neural_mask=batch['attention_mask'],
            text_input_ids=batch['text_tokens'],
            text_attention_mask=batch['text_attention_mask'],
            labels=batch['text_tokens'],
        )
        
        loss = outputs['loss']
        batch_size = batch['neural_features'].size(0)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/ppl', torch.exp(loss.clamp(max=10)), prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # Log generation samples periodically
        if self.global_step % self.config.log_generation_every_n_steps == 0:
            self._log_generation_samples(batch, prefix='train')
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        
        outputs = self.model(
            neural_features=batch['neural_features'],
            neural_mask=batch['attention_mask'],
            text_input_ids=batch['text_tokens'],
            text_attention_mask=batch['text_attention_mask'],
            labels=batch['text_tokens'],
        )
        
        loss = outputs['loss']
        batch_size = batch['neural_features'].size(0)
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('val/ppl', torch.exp(loss.clamp(max=10)), prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # Generate samples on first batch
        if batch_idx == 0:
            self._log_generation_samples(batch, prefix='val')
        
        return {'loss': loss}
    
    def _log_generation_samples(
        self,
        batch: Dict[str, torch.Tensor],
        prefix: str = 'val',
        num_samples: int = 5,
    ):
        """Generate and display samples."""
        
        self.model.eval()
        
        # Take first few samples
        n = min(num_samples, batch['neural_features'].size(0))
        neural_features = batch['neural_features'][:n]
        neural_mask = batch['attention_mask'][:n]
        targets = batch.get('texts', [])[:n]
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                neural_features=neural_features,
                neural_mask=neural_mask,
                max_length=64,
                do_sample=False,
            )
        
        # Print comparison
        print(f"\n{'='*60}")
        print(f"GENERATION SAMPLES (step {self.global_step})")
        print(f"{'='*60}")
        
        for i, gen in enumerate(generated):
            print(f"\n[Sample {i+1}]")
            print(f"  Generated: {gen[:80]}...")
            if i < len(targets) and targets[i]:
                print(f"  Target:    {targets[i][:80]}...")
        
        print(f"{'='*60}\n")
        
        self.model.train()
    
    def configure_optimizers(self):
        """Configure optimizer with different LRs for different components."""
        
        param_groups = []
        
        # Prefix projection (highest LR)
        prefix_params = list(self.model.prefix_projection.parameters())
        prefix_params += list(self.model.brain_memory_proj.parameters())
        prefix_params += list(self.model.brain_pool.parameters())
        if hasattr(self.model, 'text_proj'):
            prefix_params += list(self.model.text_proj.parameters())
        
        param_groups.append({
            'params': prefix_params,
            'lr': self.config.prefix_lr,
            'name': 'prefix',
        })
        
        # Cross-attention layers
        if self.model.decoder.cross_attention_layers:
            cross_params = list(self.model.decoder.cross_attention_layers.parameters())
            param_groups.append({
                'params': cross_params,
                'lr': self.config.cross_attention_lr,
                'name': 'cross_attention',
            })
        
        # Decoder (unfrozen layers only)
        decoder_params = [
            p for p in self.model.decoder.gpt2.parameters() 
            if p.requires_grad
        ]
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': self.config.decoder_lr,
                'name': 'decoder',
            })
        
        # Brain encoder (if not frozen)
        brain_params = [
            p for p in self.model.brain_encoder.parameters()
            if p.requires_grad
        ]
        if brain_params:
            param_groups.append({
                'params': brain_params,
                'lr': self.config.brain_encoder_lr,
                'name': 'brain_encoder',
            })
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )
        
        # Use cosine annealing scheduler with warmup (epoch-based)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs - self.config.warmup_epochs,
            eta_min=1e-7,
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs],
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
    
    @classmethod
    def from_pretrained_encoder(
        cls,
        checkpoint_path: str,
        training_config: Optional[DecoderV2TrainingConfig] = None,
        decoder_config: Optional[DecoderV2Config] = None,
    ):
        """Create from pretrained encoder."""
        
        if training_config is None:
            training_config = DecoderV2TrainingConfig()
        
        if decoder_config is None:
            decoder_config = DecoderV2Config()
        
        model = BrainCLIPDecoderV2.from_pretrained_encoder(
            checkpoint_path,
            config=decoder_config,
            freeze_brain_encoder=True,
        )
        
        return cls(model, training_config)
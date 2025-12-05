"""
BrainCLIP Lightning Module
==========================

PyTorch Lightning module for training BrainCLIP with:
- Configurable optimizers and schedulers
- Comprehensive logging
- Gradient clipping
- Mixed precision support
"""

import torch
import lightning as pl
from typing import Dict, List
from transformers import get_cosine_schedule_with_warmup

import sys
sys.path.append('..')

from models.brainclip import BrainCLIP, BrainCLIPConfig, create_brainclip_model
from losses.clip_loss import BrainCLIPLoss, compute_retrieval_metrics


class BrainCLIPLightningModule(pl.LightningModule):
    """
    Lightning module for BrainCLIP training.
    
    Handles:
    - Model forward pass
    - Loss computation
    - Optimizer and scheduler configuration
    - Logging to WandB
    - Validation metrics and sample collection
    """
    
    def __init__(
        self,
        model_config: BrainCLIPConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.03,           # Balanced
        warmup_epochs: int = 5,
        max_epochs: int = 65,
        batch_size: int = 64,
        contrastive_type: str = "clip",
        label_smoothing: float = 0.05,        # Balanced
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = BrainCLIP(model_config)
        
        # Create loss function with stronger regularization
        self.loss_fn = BrainCLIPLoss(
            contrastive_type=contrastive_type,
            temperature=model_config.temperature,
            label_smoothing=label_smoothing,
        )
        
        # Store config
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        
        # Validation sample storage for logging
        self.val_samples = {
            'brain_emb': [],
            'text_emb': [],
            'brain_emb_unnorm': [],
            'text_emb_unnorm': [],
            'texts': [],
        }
    
    def forward(
        self,
        neural_features: torch.Tensor,
        neural_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(
            neural_features=neural_features,
            neural_mask=neural_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        )
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Dictionary with:
                - neural_features: (B, T, 512)
                - attention_mask: (B, T)
                - text_tokens: (B, seq_len)
                - text_attention_mask: (B, seq_len)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Forward pass
        outputs = self(
            neural_features=batch['neural_features'],
            neural_mask=batch['attention_mask'],
            text_input_ids=batch['text_tokens'],
            text_attention_mask=batch['text_attention_mask'],
        )
        
        # Compute loss (pass unnormalized embeddings for variance regularization)
        loss_dict = self.loss_fn(
            brain_emb=outputs['brain_emb'],
            text_emb=outputs['text_emb'],
            temperature=outputs['temperature'],
            brain_emb_unnorm=outputs.get('brain_emb_unnorm'),
            text_emb_unnorm=outputs.get('text_emb_unnorm'),
        )
        
        # Log metrics
        batch_size = batch['neural_features'].size(0)
        self.log('train/loss', loss_dict['loss'], prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/loss_b2t', loss_dict['loss_b2t'], sync_dist=True, batch_size=batch_size)
        self.log('train/loss_t2b', loss_dict['loss_t2b'], sync_dist=True, batch_size=batch_size)
        self.log('train/accuracy_b2t', loss_dict['accuracy_b2t'], prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/accuracy_t2b', loss_dict['accuracy_t2b'], sync_dist=True, batch_size=batch_size)
        self.log('train/temperature', outputs['temperature'], sync_dist=True, batch_size=batch_size)
        
        # Log variance metrics if available
        if 'brain_variance' in loss_dict:
            self.log('train/brain_variance', loss_dict['brain_variance'], sync_dist=True, batch_size=batch_size)
            self.log('train/text_variance', loss_dict['text_variance'], sync_dist=True, batch_size=batch_size)
            self.log('train/variance_loss', loss_dict['variance_loss'], sync_dist=True, batch_size=batch_size)
        
        return loss_dict['loss']
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Outputs dictionary for epoch-end processing
        """
        # Forward pass
        outputs = self(
            neural_features=batch['neural_features'],
            neural_mask=batch['attention_mask'],
            text_input_ids=batch['text_tokens'],
            text_attention_mask=batch['text_attention_mask'],
        )
        
        # Compute loss (pass unnormalized embeddings for variance regularization)
        loss_dict = self.loss_fn(
            brain_emb=outputs['brain_emb'],
            text_emb=outputs['text_emb'],
            temperature=outputs['temperature'],
            brain_emb_unnorm=outputs.get('brain_emb_unnorm'),
            text_emb_unnorm=outputs.get('text_emb_unnorm'),
        )
        
        # Log metrics
        batch_size = batch['neural_features'].size(0)
        self.log('val/loss', loss_dict['loss'], prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('val/loss_b2t', loss_dict['loss_b2t'], sync_dist=True, batch_size=batch_size)
        self.log('val/loss_t2b', loss_dict['loss_t2b'], sync_dist=True, batch_size=batch_size)
        self.log('val/accuracy_b2t', loss_dict['accuracy_b2t'], prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('val/accuracy_t2b', loss_dict['accuracy_t2b'], sync_dist=True, batch_size=batch_size)
        
        # Compute retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            outputs['brain_emb'],
            outputs['text_emb'],
            k_values=(1, 5, 10),
        )
        
        for name, value in retrieval_metrics.items():
            self.log(f'val/{name}', value, sync_dist=True, batch_size=batch_size)
        
        # Store samples for end-of-epoch logging (first few batches)
        if batch_idx < 5:
            self.val_samples['brain_emb'].append(outputs['brain_emb'].detach().cpu())
            self.val_samples['text_emb'].append(outputs['text_emb'].detach().cpu())
            # Store unnormalized embeddings for variance monitoring
            if 'brain_emb_unnorm' in outputs and outputs['brain_emb_unnorm'] is not None:
                self.val_samples['brain_emb_unnorm'].append(outputs['brain_emb_unnorm'].detach().cpu())
            if 'text_emb_unnorm' in outputs and outputs['text_emb_unnorm'] is not None:
                self.val_samples['text_emb_unnorm'].append(outputs['text_emb_unnorm'].detach().cpu())
            self.val_samples['texts'].extend(batch['texts'])
        
        return {
            'loss': loss_dict['loss'],
            'brain_emb': outputs['brain_emb'],
            'text_emb': outputs['text_emb'],
            # Include unnormalized embeddings for variance monitoring (collapse detection)
            'brain_emb_unnorm': outputs.get('brain_emb_unnorm'),
            'text_emb_unnorm': outputs.get('text_emb_unnorm'),
        }
    
    def test_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step - handles both labeled and unlabeled test data.
        
        For the NEJM dataset, test set has NO ground truth text labels.
        We can only encode brain signals and store embeddings for later
        retrieval-based evaluation.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Outputs dictionary with brain embeddings
        """
        # Check if we have valid text tokens (test set may not have them)
        has_text_labels = (
            'text_tokens' in batch and 
            batch['text_tokens'] is not None and
            'texts' in batch and
            len(batch['texts']) > 0 and
            any(len(t) > 0 for t in batch['texts'])  # Check for non-empty texts
        )
        
        batch_size = batch['neural_features'].size(0)
        
        if has_text_labels:
            # Standard evaluation with ground truth text
            outputs = self(
                neural_features=batch['neural_features'],
                neural_mask=batch['attention_mask'],
                text_input_ids=batch['text_tokens'],
                text_attention_mask=batch['text_attention_mask'],
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                brain_emb=outputs['brain_emb'],
                text_emb=outputs['text_emb'],
                temperature=outputs['temperature'],
                brain_emb_unnorm=outputs.get('brain_emb_unnorm'),
                text_emb_unnorm=outputs.get('text_emb_unnorm'),
            )
            
            # Log metrics
            self.log('test/loss', loss_dict['loss'], prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log('test/loss_b2t', loss_dict['loss_b2t'], sync_dist=True, batch_size=batch_size)
            self.log('test/loss_t2b', loss_dict['loss_t2b'], sync_dist=True, batch_size=batch_size)
            self.log('test/accuracy_b2t', loss_dict['accuracy_b2t'], prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log('test/accuracy_t2b', loss_dict['accuracy_t2b'], sync_dist=True, batch_size=batch_size)
            
            # Compute retrieval metrics
            retrieval_metrics = compute_retrieval_metrics(
                outputs['brain_emb'],
                outputs['text_emb'],
                k_values=(1, 5, 10),
            )
            
            for name, value in retrieval_metrics.items():
                self.log(f'test/{name}', value, sync_dist=True, batch_size=batch_size)
            
            return {
                'loss': loss_dict['loss'],
                'brain_emb': outputs['brain_emb'],
                'text_emb': outputs['text_emb'],
            }
        else:
            # Test set without ground truth text - only encode brain signals
            # This is the case for the NEJM Brain-to-Text benchmark
            brain_emb = self.model.encode_brain(
                batch['neural_features'],
                batch['attention_mask'],
            )
            
            # Log that we're in inference-only mode
            if batch_idx == 0:
                print("\n[INFO] Test set has no ground truth text - running in inference mode")
                print("[INFO] Use inference.py to retrieve texts for test brain signals\n")
            
            # Log embedding statistics only
            self.log('test/brain_emb_norm', brain_emb.norm(dim=1).mean(), sync_dist=True, batch_size=batch_size)
            
            return {
                'brain_emb': brain_emb,
            }
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.
        
        Logs:
        - Embedding visualization
        - Retrieval examples
        - Similarity matrix
        """
        if len(self.val_samples['brain_emb']) > 0 and len(self.val_samples['texts']) > 0:
            # Aggregate embeddings
            brain_emb = torch.cat(self.val_samples['brain_emb'], dim=0)
            text_emb = torch.cat(self.val_samples['text_emb'], dim=0)
            texts = self.val_samples['texts']
            
            # Log to wandb if available
            try:
                import wandb
                if self.logger is not None and wandb.run is not None:
                    self._log_samples_to_wandb(brain_emb, text_emb, texts)
            except ImportError:
                pass
        
        # Clear samples
        self.val_samples = {
            'brain_emb': [],
            'text_emb': [],
            'brain_emb_unnorm': [],
            'text_emb_unnorm': [],
            'texts': [],
        }
    
    def _log_samples_to_wandb(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
        texts: List[str],
    ):
        """Log sample visualizations to WandB."""
        try:
            import wandb
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.manifold import TSNE
            
            # Debug info
            print(f"[DEBUG] _log_samples_to_wandb called with brain_emb={brain_emb.shape}, text_emb={text_emb.shape}, texts={len(texts)}")
            
            # Ensure we have a valid wandb run
            if wandb.run is None:
                print("Warning: No active wandb run, skipping sample logging")
                return
            
            # Validate data
            if len(brain_emb) == 0 or len(text_emb) == 0 or len(texts) == 0:
                print(f"Warning: Empty data - brain_emb={len(brain_emb)}, text_emb={len(text_emb)}, texts={len(texts)}")
                return
            
            # Ensure consistent lengths
            min_len = min(len(brain_emb), len(text_emb), len(texts))
            brain_emb = brain_emb[:min_len]
            text_emb = text_emb[:min_len]
            texts = texts[:min_len]
            
            # 1. Log embedding t-SNE visualization
            n_samples = min(100, len(brain_emb))
            
            combined_emb = torch.cat([
                brain_emb[:n_samples],
                text_emb[:n_samples]
            ], dim=0).numpy()
            
            if combined_emb.shape[0] > 5:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, combined_emb.shape[0]-1))
                emb_2d = tsne.fit_transform(combined_emb)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(
                    emb_2d[:n_samples, 0], emb_2d[:n_samples, 1],
                    c='blue', label='Brain', alpha=0.6
                )
                ax.scatter(
                    emb_2d[n_samples:, 0], emb_2d[n_samples:, 1],
                    c='red', label='Text', alpha=0.6
                )
                
                # Draw lines connecting matching pairs
                for i in range(min(10, n_samples)):
                    ax.plot(
                        [emb_2d[i, 0], emb_2d[n_samples + i, 0]],
                        [emb_2d[i, 1], emb_2d[n_samples + i, 1]],
                        'gray', alpha=0.3, linewidth=0.5
                    )
                
                ax.legend()
                ax.set_title('Brain-Text Embedding Space (t-SNE)')
                plt.tight_layout()
                
                wandb.log({'val/embedding_tsne': wandb.Image(fig)})
                plt.close(fig)
            
            # 2. Log similarity matrix
            n_sim = min(20, len(brain_emb))
            sim_matrix = (brain_emb[:n_sim] @ text_emb[:n_sim].T).numpy()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(sim_matrix, cmap='viridis', aspect='auto')
            ax.set_xlabel('Text Index')
            ax.set_ylabel('Brain Index')
            ax.set_title('Brain-Text Similarity Matrix')
            plt.colorbar(im)
            plt.tight_layout()
            
            wandb.log({'val/similarity_matrix': wandb.Image(fig)})
            plt.close(fig)
            
            # 3. Log retrieval examples as table
            sim = brain_emb @ text_emb.T
            
            # Ensure we have enough samples for top-k
            k = min(5, len(text_emb))
            if k == 0:
                print("Warning: No text embeddings for retrieval table")
                return
                
            _, indices = sim.topk(k, dim=1)
            
            columns = ['Brain Index', 'True Text', 'Top-1 Retrieved', 'Top-5 Retrieved', 'Correct']
            data = []
            
            # Use min of available samples
            n_examples = min(10, len(brain_emb), len(texts))
            print(f"[DEBUG] Creating retrieval table with {n_examples} examples, k={k}")
            
            for i in range(n_examples):
                top_k_indices = indices[i].tolist()
                # Safely get retrieved texts
                top_k_texts = []
                for j in top_k_indices:
                    if j < len(texts):
                        top_k_texts.append(texts[j][:100])  # Truncate long texts
                    else:
                        top_k_texts.append('[N/A]')
                
                is_correct = (top_k_indices[0] == i) if top_k_indices else False
                
                true_text = texts[i][:100] if i < len(texts) else '[N/A]'
                
                data.append([
                    i,
                    true_text,
                    top_k_texts[0] if top_k_texts else '[N/A]',
                    ' | '.join(top_k_texts[:3]),  # Limit to top 3 for display
                    'Yes' if is_correct else 'No',
                ])
            
            print(f"[DEBUG] Retrieval table data: {len(data)} rows")
            if len(data) > 0:
                table = wandb.Table(columns=columns, data=data)
                wandb.log({'val/retrieval_examples': table})
                print(f"[DEBUG] Successfully logged retrieval table")
            else:
                print("Warning: No data for retrieval table")
            
        except Exception as e:
            import traceback
            print(f"Warning: Failed to log samples to wandb: {e}")
            traceback.print_exc()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for different learning rates
        brain_params = list(self.model.brain_encoder.parameters()) + \
                      list(self.model.brain_projection.parameters())
        text_params = list(self.model.text_encoder.parameters()) + \
                     list(self.model.text_projection.parameters())
        
        # Different learning rates for frozen vs unfrozen
        optimizer_groups = [
            {
                'params': brain_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for p in text_params if p.requires_grad],
                'lr': self.learning_rate * 0.1,  # Lower LR for pretrained
                'weight_decay': self.weight_decay,
            },
        ]
        
        # Add temperature if learnable
        if hasattr(self.model, 'log_temperature') and self.model.log_temperature.requires_grad:
            optimizer_groups.append({
                'params': [self.model.log_temperature],
                'lr': self.learning_rate,
                'weight_decay': 0.0,  # No weight decay for temperature
            })
        
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Scheduler
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(self.warmup_epochs * num_training_steps / self.max_epochs)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
    
    def get_tokenizer(self):
        """Get the text tokenizer."""
        return self.model.get_tokenizer()


def create_lightning_module(
    brain_hidden_dim: int = 256,
    brain_num_layers: int = 4,
    text_model_name: str = "distilbert-base-uncased",
    freeze_text: bool = True,
    embedding_dim: int = 256,
    temperature: float = 0.07,
    learning_rate: float = 1e-4,
    **kwargs,
) -> BrainCLIPLightningModule:
    """
    Factory function to create Lightning module.
    
    Args:
        brain_hidden_dim: Brain encoder hidden dimension
        brain_num_layers: Number of transformer layers
        text_model_name: Pretrained text model name
        freeze_text: Whether to freeze text encoder
        embedding_dim: Shared embedding dimension
        temperature: Contrastive loss temperature
        learning_rate: Learning rate
        **kwargs: Additional arguments for LightningModule
        
    Returns:
        Configured BrainCLIPLightningModule
    """
    from models.brain_encoder import BrainEncoderConfig
    from models.text_encoder import TextEncoderConfig
    from models.brainclip import BrainCLIPConfig, ProjectionConfig
    
    brain_config = BrainEncoderConfig(
        hidden_dim=brain_hidden_dim,
        num_temporal_layers=brain_num_layers,
        num_spatiotemporal_layers=brain_num_layers,
    )
    
    text_config = TextEncoderConfig(
        model_name=text_model_name,
        freeze=freeze_text,
    )
    
    projection_config = ProjectionConfig(
        output_dim=embedding_dim,
    )
    
    model_config = BrainCLIPConfig(
        brain_encoder=brain_config,
        text_encoder=text_config,
        projection=projection_config,
        temperature=temperature,
    )
    
    return BrainCLIPLightningModule(
        model_config=model_config,
        learning_rate=learning_rate,
        **kwargs,
    )
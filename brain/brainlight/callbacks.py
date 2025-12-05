"""
BrainCLIP Callbacks
===================

Custom PyTorch Lightning callbacks for:
- WandB sample logging at validation end
- Embedding visualization
- Retrieval example logging
- Learning rate monitoring
- Model checkpointing with custom logic
"""

import torch
import numpy as np
import lightning as pl
from lightning.pytorch.callbacks import Callback
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class WandBSampleLogger(Callback):
    """
    Callback to log sample visualizations to WandB at validation end.
    
    Logs:
    1. t-SNE visualization of embedding space
    2. Similarity matrix heatmap
    3. Retrieval examples table
    4. Per-batch accuracy histogram
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        log_every_n_epochs: int = 1,
    ):
        """
        Args:
            num_samples: Number of samples to visualize
            log_every_n_epochs: Log frequency
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        
        # Storage for embeddings during validation
        self.brain_embeddings = []
        self.text_embeddings = []
        self.texts = []
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Collect embeddings from validation batches."""
        if len(self.brain_embeddings) * outputs['brain_emb'].shape[0] < self.num_samples:
            self.brain_embeddings.append(outputs['brain_emb'].detach().cpu())
            self.text_embeddings.append(outputs['text_emb'].detach().cpu())
            self.texts.extend(batch['texts'])
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """Log visualizations at the end of validation."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self._clear_storage()
            return
        
        if len(self.brain_embeddings) == 0:
            return
        
        try:
            import wandb
            
            # Concatenate all embeddings
            brain_emb = torch.cat(self.brain_embeddings, dim=0)[:self.num_samples]
            text_emb = torch.cat(self.text_embeddings, dim=0)[:self.num_samples]
            texts = self.texts[:self.num_samples]
            
            # Validate data consistency
            min_samples = min(len(brain_emb), len(text_emb), len(texts))
            if min_samples == 0:
                print(f"Warning: No valid samples for logging (brain_emb={len(brain_emb)}, text_emb={len(text_emb)}, texts={len(texts)})")
                self._clear_storage()
                return
            
            # Truncate to consistent length
            brain_emb = brain_emb[:min_samples]
            text_emb = text_emb[:min_samples]
            texts = texts[:min_samples]
            
            logger = trainer.logger
            if logger is None or not hasattr(logger, 'experiment'):
                self._clear_storage()
                return
            
            # 1. Log t-SNE visualization
            tsne_fig = self._create_tsne_plot(brain_emb, text_emb)
            if tsne_fig is not None:
                logger.experiment.log({
                    'val/embedding_tsne': wandb.Image(tsne_fig),
                    'epoch': trainer.current_epoch,
                })
                plt.close(tsne_fig)
            
            # 2. Log similarity matrix
            sim_fig = self._create_similarity_matrix(brain_emb, text_emb)
            logger.experiment.log({
                'val/similarity_matrix': wandb.Image(sim_fig),
                'epoch': trainer.current_epoch,
            })
            plt.close(sim_fig)
            
            # 3. Log retrieval examples
            table = self._create_retrieval_table(brain_emb, text_emb, texts)
            logger.experiment.log({
                'val/retrieval_examples': table,
                'epoch': trainer.current_epoch,
            })
            
            # 4. Log embedding statistics
            self._log_embedding_stats(logger, brain_emb, text_emb, trainer.current_epoch)
            
        except Exception as e:
            import traceback
            print(f"Warning: Failed to log samples: {e}")
            traceback.print_exc()
        
        self._clear_storage()
    
    def _clear_storage(self):
        """Clear stored embeddings."""
        self.brain_embeddings = []
        self.text_embeddings = []
        self.texts = []
    
    def _create_tsne_plot(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Optional[plt.Figure]:
        """Create t-SNE visualization of embedding space."""
        try:
            from sklearn.manifold import TSNE
            
            n = min(100, len(brain_emb))
            combined = torch.cat([brain_emb[:n], text_emb[:n]], dim=0).numpy()
            
            if combined.shape[0] < 5:
                return None
            
            perplexity = min(30, combined.shape[0] - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            emb_2d = tsne.fit_transform(combined)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot brain embeddings
            ax.scatter(
                emb_2d[:n, 0], emb_2d[:n, 1],
                c='steelblue', label='Brain', alpha=0.7, s=50
            )
            
            # Plot text embeddings
            ax.scatter(
                emb_2d[n:, 0], emb_2d[n:, 1],
                c='coral', label='Text', alpha=0.7, s=50
            )
            
            # Draw connections for matching pairs
            for i in range(min(20, n)):
                ax.plot(
                    [emb_2d[i, 0], emb_2d[n + i, 0]],
                    [emb_2d[i, 1], emb_2d[n + i, 1]],
                    'gray', alpha=0.3, linewidth=0.5
                )
            
            ax.legend(fontsize=12)
            ax.set_title('Brain-Text Embedding Space (t-SNE)', fontsize=14)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            print("sklearn not available for t-SNE visualization")
            return None
    
    def _create_similarity_matrix(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> plt.Figure:
        """Create similarity matrix heatmap."""
        n = min(30, len(brain_emb))
        sim = (brain_emb[:n] @ text_emb[:n].T).numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(sim, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add diagonal marker
        for i in range(n):
            ax.plot(i, i, 'k+', markersize=8)
        
        ax.set_xlabel('Text Index', fontsize=12)
        ax.set_ylabel('Brain Index', fontsize=12)
        ax.set_title('Brain-Text Similarity Matrix', fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity')
        
        plt.tight_layout()
        return fig
    
    def _create_retrieval_table(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
        texts: List[str],
    ):
        """Create WandB table with retrieval examples."""
        import wandb
        
        # Ensure we don't request more indices than available
        k = min(5, len(text_emb))
        if k == 0:
            return wandb.Table(columns=['Index', 'True Text', 'Top-1', 'Top-2', 'Top-3', 'Correct'], data=[])
        
        sim = brain_emb @ text_emb.T
        _, top5_indices = sim.topk(k, dim=1)
        
        columns = ['Index', 'True Text', 'Top-1', 'Top-2', 'Top-3', 'Correct']
        data = []
        
        n_examples = min(20, len(brain_emb), len(texts))
        for i in range(n_examples):
            indices = top5_indices[i].tolist()
            # Safely retrieve texts with boundary checking
            retrieved = []
            for j in indices[:3]:
                if j < len(texts):
                    retrieved.append(texts[j])
                else:
                    retrieved.append('[N/A]')
            # Pad if we have fewer than 3 retrieved
            while len(retrieved) < 3:
                retrieved.append('[N/A]')
            
            is_correct = indices[0] == i if indices else False
            
            data.append([
                i,
                texts[i] if i < len(texts) else '[N/A]',
                retrieved[0],
                retrieved[1],
                retrieved[2],
                'Yes' if is_correct else 'No',
            ])
        
        return wandb.Table(columns=columns, data=data)
    
    def _log_embedding_stats(
        self,
        logger,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
        epoch: int,
    ):
        """Log embedding statistics."""
        # Compute statistics
        brain_norm = brain_emb.norm(dim=1).mean().item()
        text_norm = text_emb.norm(dim=1).mean().item()
        
        # Cosine similarity for matching pairs
        diag_sim = (brain_emb * text_emb).sum(dim=1).mean().item()
        
        # Off-diagonal similarity (negative pairs)
        sim_matrix = brain_emb @ text_emb.T
        mask = ~torch.eye(len(brain_emb), dtype=bool)
        off_diag_sim = sim_matrix[mask].mean().item()
        
        logger.experiment.log({
            'val/brain_emb_norm': brain_norm,
            'val/text_emb_norm': text_norm,
            'val/positive_similarity': diag_sim,
            'val/negative_similarity': off_diag_sim,
            'val/similarity_gap': diag_sim - off_diag_sim,
            'epoch': epoch,
        })


class GradientMonitor(Callback):
    """Monitor gradient statistics during training."""
    
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
    ):
        """Log gradient statistics before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        if trainer.logger is None:
            return
        
        grad_norms = {}
        
        # Brain encoder gradients
        brain_grads = []
        for name, param in pl_module.model.brain_encoder.named_parameters():
            if param.grad is not None:
                brain_grads.append(param.grad.norm().item())
        
        if brain_grads:
            grad_norms['grad/brain_encoder_mean'] = np.mean(brain_grads)
            grad_norms['grad/brain_encoder_max'] = np.max(brain_grads)
        
        # Text encoder gradients (only unfrozen)
        text_grads = []
        for name, param in pl_module.model.text_encoder.named_parameters():
            if param.grad is not None:
                text_grads.append(param.grad.norm().item())
        
        if text_grads:
            grad_norms['grad/text_encoder_mean'] = np.mean(text_grads)
            grad_norms['grad/text_encoder_max'] = np.max(text_grads)
        
        # Projection head gradients
        proj_grads = []
        for param in pl_module.model.brain_projection.parameters():
            if param.grad is not None:
                proj_grads.append(param.grad.norm().item())
        for param in pl_module.model.text_projection.parameters():
            if param.grad is not None:
                proj_grads.append(param.grad.norm().item())
        
        if proj_grads:
            grad_norms['grad/projection_mean'] = np.mean(proj_grads)
        
        trainer.logger.log_metrics(grad_norms, step=trainer.global_step)


class TemperatureMonitor(Callback):
    """Monitor learned temperature parameter."""
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Log temperature at each step."""
        if trainer.global_step % 50 == 0 and trainer.logger is not None:
            temp = pl_module.model.temperature.item()
            trainer.logger.log_metrics(
                {'train/temperature': temp},
                step=trainer.global_step
            )


class EmbeddingDiversityCallback(Callback):
    """
    Monitor embedding diversity to detect collapse.
    
    Checks if embeddings are collapsing to a single point
    by monitoring the variance and pairwise distances.
    
    IMPORTANT: This monitors UNNORMALIZED embeddings. Normalized embeddings
    (L2-normalized) will always have low variance since they lie on a unit
    hypersphere, which is NOT indicative of collapse.
    """
    
    def __init__(self, check_every_n_epochs: int = 5):
        super().__init__()
        self.check_every_n_epochs = check_every_n_epochs
        self.embeddings = []
        self.embeddings_unnorm = []
    
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        """Collect embeddings."""
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return
        
        if batch_idx < 5:
            self.embeddings.append(outputs['brain_emb'].detach().cpu())
            # Prefer unnormalized embeddings for variance monitoring
            if 'brain_emb_unnorm' in outputs and outputs['brain_emb_unnorm'] is not None:
                self.embeddings_unnorm.append(outputs['brain_emb_unnorm'].detach().cpu())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Check for embedding collapse."""
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            self.embeddings = []
            self.embeddings_unnorm = []
            return
        
        if len(self.embeddings) == 0:
            return
        
        # Use unnormalized embeddings for variance if available, otherwise use normalized
        if len(self.embeddings_unnorm) > 0:
            embeddings_for_var = torch.cat(self.embeddings_unnorm, dim=0)
            using_unnorm = True
        else:
            embeddings_for_var = torch.cat(self.embeddings, dim=0)
            using_unnorm = False
        
        # Always use normalized embeddings for pairwise distance (more meaningful on hypersphere)
        embeddings_norm = torch.cat(self.embeddings, dim=0)
        
        # Compute variance per dimension (use unnormalized for meaningful variance)
        variance = embeddings_for_var.var(dim=0).mean().item()
        
        # Compute average pairwise distance (use normalized embeddings)
        n = min(100, len(embeddings_norm))
        sample = embeddings_norm[:n]
        distances = torch.cdist(sample, sample)
        avg_distance = distances[~torch.eye(n, dtype=bool)].mean().item()
        
        if trainer.logger is not None:
            metrics = {
                'val/embedding_variance': variance,
                'val/avg_pairwise_distance': avg_distance,
            }
            if using_unnorm:
                metrics['val/embedding_variance_unnorm'] = variance
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
        
        # Warn if variance is very low (potential collapse)
        # Threshold depends on whether using normalized or unnormalized embeddings
        collapse_threshold = 0.01 if using_unnorm else 0.001
        if variance < collapse_threshold:
            emb_type = "unnormalized" if using_unnorm else "normalized"
            print(f"WARNING: Low {emb_type} embedding variance ({variance:.6f}) - possible collapse!")
        
        # Also check pairwise distance - if very low, embeddings are too similar
        if avg_distance < 0.1:
            print(f"WARNING: Low avg pairwise distance ({avg_distance:.4f}) - embeddings too similar!")
        
        self.embeddings = []
        self.embeddings_unnorm = []


def get_callbacks(
    project_name: str = "brainclip",
    checkpoint_dir: str = "checkpoints",
    log_samples: bool = True,
    monitor_gradients: bool = True,
) -> List[Callback]:
    """
    Get list of callbacks for training.
    
    Args:
        project_name: WandB project name
        checkpoint_dir: Directory for checkpoints
        log_samples: Whether to log sample visualizations
        monitor_gradients: Whether to monitor gradients
        
    Returns:
        List of configured callbacks
    """
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    
    callbacks = []
    
    # Model checkpointing
    callbacks.append(ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='brainclip-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
        save_last=True,
    ))
    
    # Early stopping
    callbacks.append(EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min',
        verbose=True,
    ))
    
    # Learning rate monitoring
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Progress bar
    callbacks.append(RichProgressBar())
    
    # Sample logging
    if log_samples:
        callbacks.append(WandBSampleLogger(num_samples=100))
    
    # Gradient monitoring
    if monitor_gradients:
        callbacks.append(GradientMonitor())
    
    # Temperature monitoring
    callbacks.append(TemperatureMonitor())
    
    # Embedding diversity
    callbacks.append(EmbeddingDiversityCallback())
    
    return callbacks
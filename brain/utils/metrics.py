"""
BrainCLIP Utilities
===================

Utility functions for:
- Evaluation metrics
- Embedding visualization
- Retrieval evaluation
- Inference helpers
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


def compute_retrieval_metrics(
    brain_emb: torch.Tensor,
    text_emb: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute retrieval metrics for brain-text alignment.
    
    Args:
        brain_emb: (N, D) brain embeddings
        text_emb: (N, D) text embeddings
        k_values: K values for R@K computation
        
    Returns:
        Dictionary with R@K and MRR metrics
    """
    N = brain_emb.shape[0]
    device = brain_emb.device
    
    # Compute similarity matrix
    sim = brain_emb @ text_emb.T  # (N, N)
    
    # Ground truth: diagonal
    labels = torch.arange(N, device=device)
    
    metrics = {}
    
    # Brain-to-text retrieval
    _, indices_b2t = sim.topk(max(k_values), dim=1)
    
    for k in k_values:
        correct = (indices_b2t[:, :k] == labels.unsqueeze(1)).any(dim=1)
        metrics[f'R@{k}_b2t'] = correct.float().mean().item()
    
    # MRR for brain-to-text
    ranks_b2t = (sim.argsort(dim=1, descending=True) == labels.unsqueeze(1))
    ranks_b2t = ranks_b2t.nonzero()[:, 1] + 1
    metrics['MRR_b2t'] = (1.0 / ranks_b2t.float()).mean().item()
    
    # Text-to-brain retrieval
    _, indices_t2b = sim.T.topk(max(k_values), dim=1)
    
    for k in k_values:
        correct = (indices_t2b[:, :k] == labels.unsqueeze(1)).any(dim=1)
        metrics[f'R@{k}_t2b'] = correct.float().mean().item()
    
    # MRR for text-to-brain
    ranks_t2b = (sim.T.argsort(dim=1, descending=True) == labels.unsqueeze(1))
    ranks_t2b = ranks_t2b.nonzero()[:, 1] + 1
    metrics['MRR_t2b'] = (1.0 / ranks_t2b.float()).mean().item()
    
    # Average metrics
    for k in k_values:
        metrics[f'R@{k}_avg'] = (metrics[f'R@{k}_b2t'] + metrics[f'R@{k}_t2b']) / 2
    metrics['MRR_avg'] = (metrics['MRR_b2t'] + metrics['MRR_t2b']) / 2
    
    return metrics


def retrieve_texts_for_brain_signal(
    brain_emb: torch.Tensor,
    text_emb_database: torch.Tensor,
    texts: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-K texts for a brain signal.
    
    Args:
        brain_emb: (D,) or (1, D) brain embedding
        text_emb_database: (M, D) database of text embeddings
        texts: List of M text strings
        top_k: Number of results to return
        
    Returns:
        List of (text, similarity_score) tuples
    """
    if brain_emb.dim() == 1:
        brain_emb = brain_emb.unsqueeze(0)
    
    # Compute similarities
    similarities = (brain_emb @ text_emb_database.T).squeeze(0)
    
    # Get top-K
    scores, indices = similarities.topk(top_k)
    
    results = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        results.append((texts[idx], score))
    
    return results


def plot_embedding_space(
    brain_emb: torch.Tensor,
    text_emb: torch.Tensor,
    labels: Optional[List[str]] = None,
    method: str = 'tsne',
    n_samples: int = 100,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot 2D visualization of embedding space.
    
    Args:
        brain_emb: Brain embeddings
        text_emb: Text embeddings
        labels: Optional text labels for hover
        method: 'tsne' or 'pca'
        n_samples: Number of samples to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n = min(n_samples, len(brain_emb))
    
    # Combine embeddings
    combined = torch.cat([
        brain_emb[:n].cpu(),
        text_emb[:n].cpu()
    ], dim=0).numpy()
    
    # Dimensionality reduction
    if method == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = min(30, combined.shape[0] - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    emb_2d = reducer.fit_transform(combined)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot brain embeddings
    ax.scatter(
        emb_2d[:n, 0], emb_2d[:n, 1],
        c='steelblue', label='Brain', alpha=0.7, s=60
    )
    
    # Plot text embeddings
    ax.scatter(
        emb_2d[n:, 0], emb_2d[n:, 1],
        c='coral', label='Text', alpha=0.7, s=60
    )
    
    # Draw connections for matching pairs
    for i in range(min(30, n)):
        ax.plot(
            [emb_2d[i, 0], emb_2d[n + i, 0]],
            [emb_2d[i, 1], emb_2d[n + i, 1]],
            'gray', alpha=0.3, linewidth=0.5
        )
    
    ax.legend(fontsize=12)
    ax.set_title(f'Brain-Text Embedding Space ({method.upper()})', fontsize=14)
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    
    plt.tight_layout()
    return fig


def plot_similarity_matrix(
    brain_emb: torch.Tensor,
    text_emb: torch.Tensor,
    n_samples: int = 30,
    texts: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot similarity matrix heatmap.
    
    Args:
        brain_emb: Brain embeddings
        text_emb: Text embeddings
        n_samples: Number of samples to show
        texts: Optional text labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n = min(n_samples, len(brain_emb))
    
    sim = (brain_emb[:n] @ text_emb[:n].T).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(sim, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add diagonal markers
    for i in range(n):
        ax.plot(i, i, 'k+', markersize=8)
    
    ax.set_xlabel('Text Index', fontsize=12)
    ax.set_ylabel('Brain Index', fontsize=12)
    ax.set_title('Brain-Text Similarity Matrix', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity')
    
    # Add text labels if provided
    if texts is not None and n <= 20:
        truncated_texts = [t[:30] + '...' if len(t) > 30 else t for t in texts[:n]]
        ax.set_xticks(range(n))
        ax.set_xticklabels(truncated_texts, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_retrieval_results(
    brain_idx: int,
    true_text: str,
    retrieved_texts: List[Tuple[str, float]],
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot retrieval results for a single brain signal.
    
    Args:
        brain_idx: Index of brain signal
        true_text: Ground truth text
        retrieved_texts: List of (text, score) tuples
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    texts = [t for t, _ in retrieved_texts]
    scores = [s for _, s in retrieved_texts]
    
    colors = ['green' if t == true_text else 'steelblue' for t in texts]
    
    y_pos = range(len(texts))
    ax.barh(y_pos, scores, color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:50] + '...' if len(t) > 50 else t for t in texts])
    ax.set_xlabel('Similarity Score')
    ax.set_title(f'Retrieval Results for Brain Signal #{brain_idx}\nTrue: "{true_text[:50]}..."')
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig


class EmbeddingDatabase:
    """
    Database for storing and querying text embeddings.
    
    Useful for retrieval-based inference.
    """
    
    def __init__(
        self,
        embeddings: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
    ):
        self.embeddings = embeddings
        self.texts = texts or []
        self.device = 'cpu'
    
    def add(self, embeddings: torch.Tensor, texts: List[str]):
        """Add embeddings to database."""
        embeddings = embeddings.cpu()
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        
        self.texts.extend(texts)
    
    def search(
        self,
        query_emb: torch.Tensor,
        top_k: int = 10,
    ) -> List[Tuple[str, float, int]]:
        """
        Search for similar texts.
        
        Args:
            query_emb: Query embedding
            top_k: Number of results
            
        Returns:
            List of (text, score, index) tuples
        """
        if self.embeddings is None or len(self.texts) == 0:
            return []
        
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        
        query_emb = query_emb.to(self.device)
        
        # Compute similarities
        sim = query_emb @ self.embeddings.T
        scores, indices = sim.squeeze(0).topk(min(top_k, len(self.texts)))
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append((self.texts[idx], score, idx))
        
        return results
    
    def to(self, device: str):
        """Move database to device."""
        self.device = device
        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device)
        return self
    
    def save(self, path: str):
        """Save database to file."""
        torch.save({
            'embeddings': self.embeddings,
            'texts': self.texts,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'EmbeddingDatabase':
        """Load database from file."""
        data = torch.load(path)
        return cls(data['embeddings'], data['texts'])
    
    def __len__(self) -> int:
        return len(self.texts)

"""
BrainCLIP Model
===============

Complete BrainCLIP model combining:
- Brain encoder (spatiotemporal transformer)
- Text encoder (pretrained LM)
- Projection heads
- Contrastive learning framework

Inspired by:
- CLIP: Cross-modal contrastive learning
- NuCLR: Spatiotemporal transformer for neural data
- TinyCLIP: Efficient distillation and affinity mimicking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .brain_encoder import BrainEncoder, BrainEncoderConfig
from .text_encoder import TextEncoder, TextEncoderConfig, create_text_encoder


@dataclass
class ProjectionConfig:
    """Configuration for projection heads."""
    brain_input_dim: int = 1024     # hidden_dim * num_arrays
    text_input_dim: int = 768       # From text encoder
    hidden_dim: int = 512
    output_dim: int = 256           # Shared embedding dimension
    dropout: float = 0.15           # Balanced dropout
    use_batch_norm: bool = False


class ProjectionHead(nn.Module):
    """
    Projection head for mapping encoder outputs to shared embedding space.
    
    Architecture:
    Linear -> LayerNorm/BatchNorm -> GELU -> Dropout -> Linear -> L2 Normalize
    
    The projection head is crucial for contrastive learning as it:
    1. Maps different-sized encoder outputs to same dimension
    2. Provides additional capacity for learning alignment
    3. Allows encoder representations to remain task-general
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        if use_batch_norm:
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm
        
        # Initialize weights to prevent collapse
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Smaller init for final layer to prevent initial collapse
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor, return_unnormalized: bool = False) -> torch.Tensor:
        """
        Project and normalize embeddings.
        
        Args:
            x: (batch, input_dim) encoder output
            return_unnormalized: If True, return (normalized, unnormalized) tuple
            
        Returns:
            (batch, output_dim) L2-normalized projection, or tuple if return_unnormalized
        """
        x = self.fc1(x)
        x = self.dropout1(x)
        
        if self.use_batch_norm and x.shape[0] > 1:
            x = self.norm(x)
        elif not self.use_batch_norm:
            x = self.norm(x)
        
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Store unnormalized for variance regularization
        x_unnorm = x
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)
        
        if return_unnormalized:
            return x, x_unnorm
        return x


@dataclass
class BrainCLIPConfig:
    """Complete configuration for BrainCLIP model."""
    
    # Component configs
    brain_encoder: BrainEncoderConfig = None
    text_encoder: TextEncoderConfig = None
    projection: ProjectionConfig = None
    
    # Contrastive learning
    temperature: float = 0.07
    learnable_temperature: bool = True
    
    def __post_init__(self):
        if self.brain_encoder is None:
            self.brain_encoder = BrainEncoderConfig()
        if self.text_encoder is None:
            self.text_encoder = TextEncoderConfig()
        if self.projection is None:
            self.projection = ProjectionConfig()


class BrainCLIP(nn.Module):
    """
    BrainCLIP: Contrastive Learning for Brain-to-Text.
    
    Learns to align neural signal embeddings with text embeddings
    in a shared latent space using contrastive learning (CLIP-style).
    
    Architecture:
        Brain Signal → Brain Encoder → Brain Projection → Shared Space
        Text         → Text Encoder  → Text Projection  → Shared Space
    
    Training:
        Contrastive loss pulls matching (brain, text) pairs together
        while pushing non-matching pairs apart.
    """
    
    def __init__(self, config: BrainCLIPConfig):
        super().__init__()
        self.config = config
        
        # Brain encoder
        self.brain_encoder = BrainEncoder(config.brain_encoder)
        
        # Text encoder
        self.text_encoder = create_text_encoder(config.text_encoder)
        
        # Update projection config with actual dimensions
        config.projection.brain_input_dim = self.brain_encoder.output_dim
        config.projection.text_input_dim = self.text_encoder.output_dim
        
        # Projection heads
        self.brain_projection = ProjectionHead(
            input_dim=config.projection.brain_input_dim,
            hidden_dim=config.projection.hidden_dim,
            output_dim=config.projection.output_dim,
            dropout=config.projection.dropout,
        )
        
        self.text_projection = ProjectionHead(
            input_dim=config.projection.text_input_dim,
            hidden_dim=config.projection.hidden_dim,
            output_dim=config.projection.output_dim,
            dropout=config.projection.dropout,
        )
        
        # Temperature parameter
        if config.learnable_temperature:
            # Initialize log temperature (more stable optimization)
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(config.temperature))
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.log(torch.tensor(config.temperature))
            )
        
        # Embedding dimension
        self.embedding_dim = config.projection.output_dim
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature (from log for stability). Clamped to prevent collapse."""
        # Clamp temperature between 0.01 and 1.0 to prevent collapse
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)
    
    def encode_brain(
        self,
        neural_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_unnormalized: bool = False,
    ) -> torch.Tensor:
        """
        Encode brain signals to embeddings.
        
        Args:
            neural_features: (batch, time, 512) neural features
            attention_mask: (batch, time) mask for valid time bins
            return_unnormalized: If True, also return unnormalized embeddings
            
        Returns:
            (batch, embedding_dim) brain embedding, or tuple if return_unnormalized
        """
        brain_hidden = self.brain_encoder(neural_features, attention_mask)
        return self.brain_projection(brain_hidden, return_unnormalized=return_unnormalized)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_unnormalized: bool = False,
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            return_unnormalized: If True, also return unnormalized embeddings
            
        Returns:
            (batch, embedding_dim) text embedding, or tuple if return_unnormalized
        """
        text_hidden = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_hidden, return_unnormalized=return_unnormalized)
    
    def forward(
        self,
        neural_features: torch.Tensor,
        neural_mask: Optional[torch.Tensor],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing embeddings and similarity matrix.
        
        Args:
            neural_features: (batch, time, 512) neural features
            neural_mask: (batch, time) neural attention mask
            text_input_ids: (batch, seq_len) text token IDs
            text_attention_mask: (batch, seq_len) text attention mask
            
        Returns:
            Dictionary with:
                - brain_emb: (batch, embedding_dim) brain embeddings
                - text_emb: (batch, embedding_dim) text embeddings
                - brain_emb_unnorm: (batch, embedding_dim) unnormalized brain embeddings
                - text_emb_unnorm: (batch, embedding_dim) unnormalized text embeddings
                - logits_per_brain: (batch, batch) similarity scores
                - logits_per_text: (batch, batch) similarity scores
                - temperature: scalar temperature value
        """
        # Encode both modalities (get both normalized and unnormalized)
        brain_emb, brain_emb_unnorm = self.encode_brain(neural_features, neural_mask, return_unnormalized=True)
        text_emb, text_emb_unnorm = self.encode_text(text_input_ids, text_attention_mask, return_unnormalized=True)
        
        # Compute similarity matrix (cosine similarity, scaled by temperature)
        # Since embeddings are L2-normalized, dot product = cosine similarity
        logits_per_brain = brain_emb @ text_emb.T / self.temperature
        logits_per_text = logits_per_brain.T
        
        return {
            "brain_emb": brain_emb,
            "text_emb": text_emb,
            "brain_emb_unnorm": brain_emb_unnorm,
            "text_emb_unnorm": text_emb_unnorm,
            "logits_per_brain": logits_per_brain,
            "logits_per_text": logits_per_text,
            "temperature": self.temperature,
        }
    
    def get_similarity(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between brain and text embeddings.
        
        Args:
            brain_emb: (N, embedding_dim) brain embeddings
            text_emb: (M, embedding_dim) text embeddings
            
        Returns:
            (N, M) similarity scores
        """
        return brain_emb @ text_emb.T
    
    def get_tokenizer(self):
        """Get the text tokenizer."""
        return self.text_encoder.get_tokenizer()


class BrainCLIPWithDistillation(BrainCLIP):
    """
    BrainCLIP with teacher model distillation (TinyCLIP-inspired).
    
    Adds affinity mimicking loss that encourages the student model
    to match the cross-modal affinity patterns of a teacher model.
    """
    
    def __init__(
        self,
        config: BrainCLIPConfig,
        teacher_model: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        
        self.teacher_model = teacher_model
        
        if teacher_model is not None:
            # Freeze teacher
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
    
    def compute_affinity_loss(
        self,
        student_brain_emb: torch.Tensor,
        student_text_emb: torch.Tensor,
        teacher_brain_emb: torch.Tensor,
        teacher_text_emb: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute affinity mimicking loss.
        
        Encourages student affinity matrix to match teacher affinity matrix.
        
        Args:
            student_brain_emb: Student brain embeddings
            student_text_emb: Student text embeddings
            teacher_brain_emb: Teacher brain embeddings
            teacher_text_emb: Teacher text embeddings
            temperature: Softmax temperature
            
        Returns:
            Affinity mimicking loss
        """
        # Compute affinity matrices (softmax over similarity)
        student_affinity_b2t = F.softmax(
            student_brain_emb @ student_text_emb.T / temperature, dim=-1
        )
        teacher_affinity_b2t = F.softmax(
            teacher_brain_emb @ teacher_text_emb.T / temperature, dim=-1
        )
        
        student_affinity_t2b = F.softmax(
            student_text_emb @ student_brain_emb.T / temperature, dim=-1
        )
        teacher_affinity_t2b = F.softmax(
            teacher_text_emb @ teacher_brain_emb.T / temperature, dim=-1
        )
        
        # KL divergence loss
        loss_b2t = F.kl_div(
            student_affinity_b2t.log(),
            teacher_affinity_b2t,
            reduction="batchmean"
        )
        loss_t2b = F.kl_div(
            student_affinity_t2b.log(),
            teacher_affinity_t2b,
            reduction="batchmean"
        )
        
        return (loss_b2t + loss_t2b) / 2


def create_brainclip_model(
    brain_config: Optional[BrainEncoderConfig] = None,
    text_model_name: str = "distilbert-base-uncased",
    freeze_text: bool = True,
    embedding_dim: int = 256,
    temperature: float = 0.07,
) -> BrainCLIP:
    """
    Factory function to create BrainCLIP model with common configurations.
    
    Args:
        brain_config: Brain encoder configuration (uses defaults if None)
        text_model_name: Name of pretrained text model
        freeze_text: Whether to freeze text encoder
        embedding_dim: Dimension of shared embedding space
        temperature: Contrastive loss temperature
        
    Returns:
        Configured BrainCLIP model
    """
    if brain_config is None:
        brain_config = BrainEncoderConfig()
    
    text_config = TextEncoderConfig(
        model_name=text_model_name,
        freeze=freeze_text,
    )
    
    projection_config = ProjectionConfig(
        output_dim=embedding_dim,
    )
    
    config = BrainCLIPConfig(
        brain_encoder=brain_config,
        text_encoder=text_config,
        projection=projection_config,
        temperature=temperature,
    )
    
    return BrainCLIP(config)


# Test
if __name__ == "__main__":
    # Create model
    model = create_brainclip_model()
    
    # Test inputs
    batch_size = 4
    seq_len = 200
    
    neural_features = torch.randn(batch_size, seq_len, 512)
    neural_mask = torch.ones(batch_size, seq_len)
    
    # Get tokenizer and create dummy text inputs
    tokenizer = model.get_tokenizer()
    texts = ["hello world", "this is a test", "brain to text", "neural signals"]
    encoding = tokenizer(texts, padding=True, return_tensors="pt")
    
    # Forward pass
    outputs = model(
        neural_features=neural_features,
        neural_mask=neural_mask,
        text_input_ids=encoding["input_ids"],
        text_attention_mask=encoding["attention_mask"],
    )
    
    print(f"Brain embedding shape: {outputs['brain_emb'].shape}")
    print(f"Text embedding shape: {outputs['text_emb'].shape}")
    print(f"Logits shape: {outputs['logits_per_brain'].shape}")
    print(f"Temperature: {outputs['temperature'].item():.4f}")
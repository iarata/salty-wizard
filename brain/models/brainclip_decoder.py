"""
BrainCLIP Decoder V2 - Stronger Brain Conditioning
===================================================

The original prefix-tuning approach fails because GPT-2 ignores the prefix.
This version uses multiple techniques to force the decoder to attend to brain signals:

1. Cross-attention layers (not just prefix)
2. Longer prefix (32 tokens instead of 8)
3. Auxiliary contrastive loss 
4. Gated fusion of brain and text representations
5. Brain-conditioned output projection

Architecture:
    Brain Signal → Encoder → Cross-Attention into Decoder → Generated Text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
)
import math

from brainlight.module import BrainCLIPLightningModule

@dataclass
class DecoderV2Config:
    """Configuration for improved decoder."""
    # Decoder model
    decoder_model_name: str = "gpt2"
    freeze_decoder_embeddings: bool = True
    freeze_decoder_layers: int = 6  # Freeze first N layers
    
    # Brain conditioning - INCREASED
    prefix_length: int = 32           # Increased from 8
    num_brain_tokens: int = 16        # Additional "brain memory" tokens
    prefix_hidden_dim: int = 768      # Match GPT-2 hidden dim
    prefix_num_layers: int = 4        # Deeper prefix projection
    
    # Cross-attention
    use_cross_attention: bool = True
    cross_attention_layers: List[int] = None  # Which layers get cross-attn
    
    # Brain encoder
    brain_embed_dim: int = 1024
    
    # Regularization
    dropout: float = 0.2              # Dropout rate
    label_smoothing: float = 0.1      # Label smoothing for CE loss
    
    # Auxiliary losses
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.1
    
    # Generation
    max_length: int = 128
    num_beams: int = 5
    repetition_penalty: float = 1.2
    
    def __post_init__(self):
        if self.cross_attention_layers is None:
            # Add cross-attention to layers 2, 4, 6, 8, 10 (every other layer)
            self.cross_attention_layers = [2, 4, 6, 8, 10]


class DeepPrefixProjection(nn.Module):
    """
    Deeper prefix projection for stronger brain signal.
    
    Projects brain embedding to prefix tokens using a multi-layer
    transformer, not just an MLP.
    """
    
    def __init__(
        self,
        brain_embed_dim: int,
        decoder_embed_dim: int,
        prefix_length: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.prefix_length = prefix_length
        self.decoder_embed_dim = decoder_embed_dim
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(brain_embed_dim, decoder_embed_dim),
            nn.LayerNorm(decoder_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Learnable prefix queries
        self.prefix_queries = nn.Parameter(
            torch.randn(prefix_length, decoder_embed_dim) * 0.02
        )
        
        # Cross-attention layers to build prefix from brain embedding
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=decoder_embed_dim,
                nhead=num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(decoder_embed_dim)
    
    def forward(self, brain_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project brain embedding to prefix tokens.
        
        Args:
            brain_embedding: (batch, brain_embed_dim)
            
        Returns:
            prefix: (batch, prefix_length, decoder_embed_dim)
        """
        batch_size = brain_embedding.size(0)
        
        # Project brain embedding
        brain_hidden = self.input_proj(brain_embedding)  # (batch, embed_dim)
        brain_hidden = brain_hidden.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Expand prefix queries for batch
        prefix = self.prefix_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attend from prefix queries to brain embedding
        for layer in self.layers:
            prefix = layer(prefix, brain_hidden)
        
        prefix = self.output_norm(prefix)
        
        return prefix


class BrainCrossAttention(nn.Module):
    """
    Cross-attention layer for injecting brain information into decoder.
    
    Added to specific decoder layers to allow direct brain conditioning.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Parameter(torch.zeros(1))  # Learnable gate, starts at 0
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        brain_memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention to brain memory.
        
        Args:
            hidden_states: (batch, seq_len, embed_dim) decoder hidden states
            brain_memory: (batch, memory_len, embed_dim) brain representations
            attention_mask: Optional mask
            
        Returns:
            Updated hidden states
        """
        # Cross-attention
        attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=brain_memory,
            value=brain_memory,
            key_padding_mask=attention_mask,
        )
        
        # Gated residual (gate starts at 0, learns to incorporate brain info)
        gate = torch.sigmoid(self.gate)
        hidden_states = self.norm(hidden_states + gate * attn_output)
        
        return hidden_states


class BrainConditionedGPT2(nn.Module):
    """
    GPT-2 with brain cross-attention layers inserted.
    """
    
    def __init__(
        self,
        config: DecoderV2Config,
    ):
        super().__init__()
        
        self.config = config
        
        # Load base GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.decoder_model_name)
        self.embed_dim = self.gpt2.config.n_embd
        self.label_smoothing = config.label_smoothing
        
        # Freeze specified layers
        if config.freeze_decoder_embeddings:
            for param in self.gpt2.transformer.wte.parameters():
                param.requires_grad = False
            for param in self.gpt2.transformer.wpe.parameters():
                param.requires_grad = False
        
        for i in range(config.freeze_decoder_layers):
            if i < len(self.gpt2.transformer.h):
                for param in self.gpt2.transformer.h[i].parameters():
                    param.requires_grad = False
        
        # Add cross-attention layers
        if config.use_cross_attention:
            self.cross_attention_layers = nn.ModuleDict()
            for layer_idx in config.cross_attention_layers:
                if layer_idx < len(self.gpt2.transformer.h):
                    self.cross_attention_layers[str(layer_idx)] = BrainCrossAttention(
                        embed_dim=self.embed_dim,
                        num_heads=8,
                        dropout=config.dropout,
                    )
        else:
            self.cross_attention_layers = None
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        brain_memory: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        """Forward pass with brain cross-attention."""
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.gpt2.transformer.wte(input_ids)
        
        # Add position embeddings
        position_ids = torch.arange(
            inputs_embeds.size(1), 
            device=inputs_embeds.device
        ).unsqueeze(0)
        position_embeds = self.gpt2.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.gpt2.transformer.drop(hidden_states)
        
        # Process through transformer layers with cross-attention
        for i, block in enumerate(self.gpt2.transformer.h):
            hidden_states = block(hidden_states)[0]
            
            # Apply brain cross-attention at specified layers
            if (self.cross_attention_layers is not None and 
                str(i) in self.cross_attention_layers and
                brain_memory is not None):
                hidden_states = self.cross_attention_layers[str(i)](
                    hidden_states, brain_memory
                )
        
        hidden_states = self.gpt2.transformer.ln_f(hidden_states)
        
        # LM head
        logits = self.gpt2.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
        
        return {'loss': loss, 'logits': logits, 'hidden_states': hidden_states}


class BrainCLIPDecoderV2(nn.Module):
    """
    Improved BrainCLIP decoder with stronger brain conditioning.
    """
    
    def __init__(
        self,
        brain_encoder: nn.Module,
        config: DecoderV2Config,
    ):
        super().__init__()
        
        self.config = config
        self.brain_encoder = brain_encoder
        
        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get decoder embed dim
        gpt2_config = GPT2Config.from_pretrained(config.decoder_model_name)
        decoder_embed_dim = gpt2_config.n_embd
        
        # Deep prefix projection
        self.prefix_projection = DeepPrefixProjection(
            brain_embed_dim=config.brain_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            prefix_length=config.prefix_length,
            num_layers=config.prefix_num_layers,
            dropout=config.dropout,
        )
        
        # Brain memory projection (for cross-attention)
        self.brain_memory_proj = nn.Sequential(
            nn.Linear(config.brain_embed_dim, decoder_embed_dim),
            nn.LayerNorm(decoder_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(decoder_embed_dim, decoder_embed_dim * config.num_brain_tokens),
        )
        self.num_brain_tokens = config.num_brain_tokens
        
        # Brain-conditioned decoder
        self.decoder = BrainConditionedGPT2(config)
        
        # Brain pooling
        brain_hidden = brain_encoder.config.hidden_dim * brain_encoder.config.num_arrays
        self.brain_pool = nn.Sequential(
            nn.Linear(brain_hidden, config.brain_embed_dim),
            nn.LayerNorm(config.brain_embed_dim),
            nn.GELU(),
        )
        
        # For auxiliary contrastive loss
        if config.use_contrastive_loss:
            self.text_proj = nn.Linear(decoder_embed_dim, config.brain_embed_dim)
    
    def encode_brain(
        self,
        neural_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode brain signals."""
        encoder_output = self.brain_encoder(neural_features, attention_mask)
        brain_embedding = self.brain_pool(encoder_output)
        return brain_embedding
    
    def forward(
        self,
        neural_features: torch.Tensor,
        neural_mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        
        batch_size = neural_features.size(0)
        device = neural_features.device
        
        # Encode brain
        brain_embedding = self.encode_brain(neural_features, neural_mask)
        
        # Get prefix tokens
        prefix_embeds = self.prefix_projection(brain_embedding)
        
        # Get brain memory for cross-attention
        brain_memory = self.brain_memory_proj(brain_embedding)
        brain_memory = brain_memory.view(
            batch_size, self.num_brain_tokens, -1
        )
        
        # Get text embeddings
        text_embeds = self.decoder.gpt2.transformer.wte(text_input_ids)
        
        # Concatenate prefix + text
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        
        # Create attention mask
        prefix_mask = torch.ones(batch_size, self.config.prefix_length, device=device)
        if text_attention_mask is not None:
            combined_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Prepare labels
        if labels is None:
            labels = text_input_ids.clone()
        
        # Pad labels for prefix
        prefix_labels = torch.full(
            (batch_size, self.config.prefix_length),
            -100, device=device, dtype=labels.dtype
        )
        combined_labels = torch.cat([prefix_labels, labels], dim=1)
        
        # Forward through decoder with cross-attention
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            brain_memory=brain_memory,
            labels=combined_labels,
        )
        
        loss = outputs['loss']
        
        # Auxiliary contrastive loss
        if self.config.use_contrastive_loss and self.training:
            # Get mean text representation (excluding prefix)
            text_hidden = outputs['hidden_states'][:, self.config.prefix_length:, :]
            if text_attention_mask is not None:
                mask = text_attention_mask.unsqueeze(-1)
                text_repr = (text_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                text_repr = text_hidden.mean(dim=1)
            
            # Project to brain space
            text_proj = self.text_proj(text_repr)
            text_proj = F.normalize(text_proj, p=2, dim=-1)
            brain_norm = F.normalize(brain_embedding, p=2, dim=-1)
            
            # Contrastive loss
            logits = brain_norm @ text_proj.T / 0.07
            targets = torch.arange(batch_size, device=device)
            contrastive_loss = F.cross_entropy(logits, targets)
            
            loss = loss + self.config.contrastive_weight * contrastive_loss
        
        return {
            'loss': loss,
            'logits': outputs['logits'],
            'brain_embedding': brain_embedding,
        }
    
    @torch.no_grad()
    def generate(
        self,
        neural_features: torch.Tensor,
        neural_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        """Generate text from brain signals."""
        
        batch_size = neural_features.size(0)
        device = neural_features.device
        max_length = max_length or self.config.max_length
        num_beams = num_beams or self.config.num_beams
        
        # Encode brain
        brain_embedding = self.encode_brain(neural_features, neural_mask)
        
        # Get prefix
        prefix_embeds = self.prefix_projection(brain_embedding)
        
        # Get brain memory
        brain_memory = self.brain_memory_proj(brain_embedding)
        brain_memory = brain_memory.view(batch_size, self.num_brain_tokens, -1)
        
        # Generate autoregressively
        generated_ids = []
        
        for i in range(batch_size):
            # Single sample
            single_prefix = prefix_embeds[i:i+1]
            single_memory = brain_memory[i:i+1]
            
            # Start with just prefix
            current_embeds = single_prefix
            generated = []
            
            for _ in range(max_length):
                # Forward pass
                outputs = self.decoder(
                    inputs_embeds=current_embeds,
                    brain_memory=single_memory,
                )
                
                # Get next token logits
                next_logits = outputs['logits'][:, -1, :] / temperature
                
                # Sample or argmax
                if do_sample:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                
                generated.append(next_token.item())
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append token embedding
                next_embed = self.decoder.gpt2.transformer.wte(next_token)
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            
            generated_ids.append(generated)
        
        # Decode
        texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text.strip())
        
        return texts
    
    @classmethod
    def from_pretrained_encoder(
        cls,
        checkpoint_path: str,
        config: Optional[DecoderV2Config] = None,
        freeze_brain_encoder: bool = True,
    ):
        """Create from pretrained BrainCLIP checkpoint."""
        
        # Load checkpoint (weights_only=False needed for custom config classes)
        pretrained = BrainCLIPLightningModule.load_from_checkpoint(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        
        brain_encoder = pretrained.model.brain_encoder
        
        if freeze_brain_encoder:
            for param in brain_encoder.parameters():
                param.requires_grad = False
        
        if config is None:
            config = DecoderV2Config()
        
        # Update brain dim
        config.brain_embed_dim = (
            brain_encoder.config.hidden_dim * brain_encoder.config.num_arrays
        )
        
        return cls(brain_encoder, config)
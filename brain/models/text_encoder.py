"""
BrainCLIP Text Encoder Module
=============================

Wrapper for pretrained text encoders (DistilBERT, CLIP, etc.)
with optional freezing and fine-tuning of specific layers.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
)
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TextEncoderConfig:
    """Configuration for text encoder."""
    
    model_name: str = "distilbert-base-uncased"
    freeze: bool = True
    freeze_layers: int = -2  # Negative = unfreeze last N layers
    max_length: int = 128
    pool_type: str = "cls"   # "cls", "mean", or "last"


class TextEncoder(nn.Module):
    """
    Text encoder wrapper for pretrained language models.
    
    Supports:
    - DistilBERT
    - BERT
    - CLIP text encoder
    - Sentence-BERT
    
    Features:
    - Selective layer freezing
    - Multiple pooling strategies
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Get output dimension
        self.output_dim = self.model.config.hidden_size
        
        # Apply freezing
        if config.freeze:
            self._freeze_parameters()
    
    def _freeze_parameters(self):
        """Freeze model parameters with optional layer unfreezing."""
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze specific layers if requested
        if self.config.freeze_layers < 0:
            # Unfreeze last N layers
            num_layers = len(self.model.encoder.layer) if hasattr(self.model, 'encoder') else \
                         len(self.model.transformer.layer) if hasattr(self.model, 'transformer') else 0
            
            if num_layers > 0:
                layers_to_unfreeze = list(range(
                    num_layers + self.config.freeze_layers,
                    num_layers
                ))
                
                encoder = self.model.encoder if hasattr(self.model, 'encoder') else \
                          self.model.transformer
                
                for layer_idx in layers_to_unfreeze:
                    for param in encoder.layer[layer_idx].parameters():
                        param.requires_grad = True
    
    def _pool_output(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states to single vector.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            
        Returns:
            (batch, hidden_dim) pooled representation
        """
        if self.config.pool_type == "cls":
            # Use CLS token (first token)
            return hidden_states[:, 0, :]
        
        elif self.config.pool_type == "mean":
            # Mean pooling over non-padded tokens
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.config.pool_type == "last":
            # Use last non-padded token
            batch_size = hidden_states.shape[0]
            last_indices = attention_mask.sum(dim=1) - 1
            last_indices = last_indices.long().clamp(min=0)
            return hidden_states[torch.arange(batch_size), last_indices]
        
        else:
            raise ValueError(f"Unknown pool type: {self.config.pool_type}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            
        Returns:
            (batch, hidden_dim) text embedding
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        hidden_states = outputs.last_hidden_state
        pooled = self._pool_output(hidden_states, attention_mask)
        
        return pooled
    
    def get_tokenizer(self):
        """Return the tokenizer for this encoder."""
        return self.tokenizer
    
    def encode_text(
        self,
        texts: list[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method to encode raw text strings.
        
        Args:
            texts: List of text strings
            device: Device to place tensors on
            
        Returns:
            (batch, hidden_dim) text embeddings
        """
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        
        if device is not None:
            encoding = {k: v.to(device) for k, v in encoding.items()}
        
        return self.forward(encoding["input_ids"], encoding["attention_mask"])


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder wrapper.
    
    Uses the text encoder from a pretrained CLIP model,
    which is specifically designed for cross-modal alignment.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
    ):
        super().__init__()
        
        from transformers import CLIPTextModel, CLIPTokenizer
        
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        CLIP uses the [EOS] token embedding as the sequence representation.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # CLIP pools from the [EOS] token
        return outputs.pooler_output
    
    def get_tokenizer(self):
        return self.tokenizer


class TinyCLIPTextEncoder(nn.Module):
    """
    TinyCLIP text encoder wrapper.
    
    Uses the distilled TinyCLIP text encoder for efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M",
        freeze: bool = True,
    ):
        super().__init__()
        
        from transformers import CLIPModel, CLIPProcessor
        
        full_model = CLIPModel.from_pretrained(model_name)
        self.model = full_model.text_model
        self.tokenizer = CLIPProcessor.from_pretrained(model_name).tokenizer
        self.output_dim = full_model.config.text_config.hidden_size
        
        # Text projection from CLIP
        self.text_projection = full_model.text_projection
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning projected text embeddings."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Pool from [EOS] token
        pooled = outputs.pooler_output
        
        # Apply text projection
        projected = self.text_projection(pooled)
        
        return projected
    
    def get_tokenizer(self):
        return self.tokenizer


def create_text_encoder(config: TextEncoderConfig) -> nn.Module:
    """
    Factory function to create appropriate text encoder.
    
    Args:
        config: Text encoder configuration
        
    Returns:
        Text encoder module
    """
    model_name = config.model_name.lower()
    
    if "clip" in model_name and "tiny" in model_name:
        return TinyCLIPTextEncoder(config.model_name, config.freeze)
    elif "clip" in model_name:
        return CLIPTextEncoder(config.model_name, config.freeze)
    else:
        return TextEncoder(config)

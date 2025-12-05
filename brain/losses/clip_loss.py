"""
BrainCLIP Loss Functions
========================

Contrastive loss functions for cross-modal learning:
- CLIP-style InfoNCE loss
- Affinity mimicking loss (TinyCLIP-inspired)
- Auxiliary losses (phoneme prediction, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CLIPLoss(nn.Module):
    """
    CLIP-style contrastive loss (InfoNCE).
    
    Symmetric cross-entropy loss over similarity matrix:
    - Brain-to-text: For each brain embedding, classify correct text
    - Text-to-brain: For each text embedding, classify correct brain signal
    
    Mathematical formulation:
    For batch of N (brain, text) pairs with similarity matrix S[i,j]:
    
    L_b2t = -1/N * Σ_i log(exp(S[i,i]/τ) / Σ_j exp(S[i,j]/τ))
    L_t2b = -1/N * Σ_j log(exp(S[j,j]/τ) / Σ_i exp(S[i,j]/τ))
    L = (L_b2t + L_t2b) / 2
    
    where τ is temperature parameter.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        use_hard_negatives: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.use_hard_negatives = use_hard_negatives
    
    def forward(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CLIP loss.
        
        Args:
            brain_emb: (N, D) L2-normalized brain embeddings
            text_emb: (N, D) L2-normalized text embeddings
            temperature: Optional temperature override
            
        Returns:
            Dictionary with:
                - loss: Total loss
                - loss_b2t: Brain-to-text loss
                - loss_t2b: Text-to-brain loss
                - accuracy_b2t: Brain-to-text retrieval accuracy
                - accuracy_t2b: Text-to-brain retrieval accuracy
        """
        if temperature is None:
            temperature = self.temperature
        
        batch_size = brain_emb.shape[0]
        device = brain_emb.device
        
        # Compute similarity matrix
        # Since embeddings are L2-normalized, dot product = cosine similarity
        logits = brain_emb @ text_emb.T / temperature  # (N, N)
        
        # Labels: diagonal entries are positive pairs
        labels = torch.arange(batch_size, device=device)
        
        # Cross-entropy loss in both directions
        if self.label_smoothing > 0:
            # Soft labels for regularization
            loss_b2t = self._cross_entropy_with_smoothing(
                logits, labels, self.label_smoothing
            )
            loss_t2b = self._cross_entropy_with_smoothing(
                logits.T, labels, self.label_smoothing
            )
        else:
            loss_b2t = F.cross_entropy(logits, labels)
            loss_t2b = F.cross_entropy(logits.T, labels)
        
        loss = (loss_b2t + loss_t2b) / 2
        
        # Compute accuracies
        with torch.no_grad():
            pred_b2t = logits.argmax(dim=1)
            pred_t2b = logits.T.argmax(dim=1)
            acc_b2t = (pred_b2t == labels).float().mean()
            acc_t2b = (pred_t2b == labels).float().mean()
        
        return {
            "loss": loss,
            "loss_b2t": loss_b2t,
            "loss_t2b": loss_t2b,
            "accuracy_b2t": acc_b2t,
            "accuracy_t2b": acc_t2b,
        }
    
    def _cross_entropy_with_smoothing(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        smoothing: float,
    ) -> torch.Tensor:
        """Cross-entropy with label smoothing."""
        num_classes = logits.shape[1]
        
        # Create smoothed labels
        smooth_labels = torch.full_like(
            logits, smoothing / (num_classes - 1)
        )
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        
        return loss


class DecoupledContrastiveLoss(nn.Module):
    """
    Decoupled Contrastive Loss (DCL).
    
    Removes positive pair similarity from denominator for better
    performance with small batch sizes (as used in NuCLR).
    
    L = -log(exp(s_pos/τ) / Σ_{neg} exp(s_neg/τ))
    
    This prevents the positive pair from competing with itself
    in the softmax denominator.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DCL loss.
        
        Args:
            brain_emb: (N, D) brain embeddings
            text_emb: (N, D) text embeddings
            temperature: Optional temperature override
            
        Returns:
            Loss dictionary
        """
        if temperature is None:
            temperature = self.temperature
        
        batch_size = brain_emb.shape[0]
        device = brain_emb.device
        
        # Similarity matrix
        sim = brain_emb @ text_emb.T / temperature
        
        # Positive similarities (diagonal)
        pos_sim = torch.diag(sim)
        
        # Create mask to exclude diagonal
        mask = ~torch.eye(batch_size, dtype=bool, device=device)
        
        # Brain-to-text loss
        neg_sim_b2t = sim.masked_select(mask).reshape(batch_size, -1)
        loss_b2t = -pos_sim + torch.logsumexp(neg_sim_b2t, dim=1)
        
        # Text-to-brain loss
        neg_sim_t2b = sim.T.masked_select(mask).reshape(batch_size, -1)
        loss_t2b = -pos_sim + torch.logsumexp(neg_sim_t2b, dim=1)
        
        loss = (loss_b2t.mean() + loss_t2b.mean()) / 2
        
        # Accuracies
        with torch.no_grad():
            labels = torch.arange(batch_size, device=device)
            acc_b2t = (sim.argmax(dim=1) == labels).float().mean()
            acc_t2b = (sim.T.argmax(dim=1) == labels).float().mean()
        
        return {
            "loss": loss,
            "loss_b2t": loss_b2t.mean(),
            "loss_t2b": loss_t2b.mean(),
            "accuracy_b2t": acc_b2t,
            "accuracy_t2b": acc_t2b,
        }


class AffinityMimickingLoss(nn.Module):
    """
    Affinity Mimicking Loss (from TinyCLIP).
    
    Distills cross-modal alignment knowledge from teacher to student
    by matching their affinity matrices (softmax of similarity).
    
    This captures not just which pairs should be similar, but the
    relative similarity structure across all pairs.
    """
    
    def __init__(
        self,
        teacher_temperature: float = 1.0,
        student_temperature: float = 1.0,
    ):
        super().__init__()
        self.teacher_temp = teacher_temperature
        self.student_temp = student_temperature
    
    def forward(
        self,
        student_brain_emb: torch.Tensor,
        student_text_emb: torch.Tensor,
        teacher_brain_emb: torch.Tensor,
        teacher_text_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute affinity mimicking loss.
        
        Args:
            student_brain_emb: Student model brain embeddings
            student_text_emb: Student model text embeddings
            teacher_brain_emb: Teacher model brain embeddings
            teacher_text_emb: Teacher model text embeddings
            
        Returns:
            Loss dictionary
        """
        # Student affinities
        student_sim = student_brain_emb @ student_text_emb.T
        student_affinity_b2t = F.softmax(student_sim / self.student_temp, dim=1)
        student_affinity_t2b = F.softmax(student_sim.T / self.student_temp, dim=1)
        
        # Teacher affinities (no gradient)
        with torch.no_grad():
            teacher_sim = teacher_brain_emb @ teacher_text_emb.T
            teacher_affinity_b2t = F.softmax(teacher_sim / self.teacher_temp, dim=1)
            teacher_affinity_t2b = F.softmax(teacher_sim.T / self.teacher_temp, dim=1)
        
        # KL divergence losses
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
        
        loss = (loss_b2t + loss_t2b) / 2
        
        return {
            "loss": loss,
            "loss_b2t": loss_b2t,
            "loss_t2b": loss_t2b,
        }


class BrainCLIPLoss(nn.Module):
    """
    Combined loss for BrainCLIP training.
    
    Combines:
    - Contrastive loss (CLIP or DCL)
    - Optional affinity mimicking (if teacher provided)
    - Variance regularization to prevent embedding collapse
    - Covariance regularization (VICReg-style)
    - Optional auxiliary losses
    """
    
    def __init__(
        self,
        contrastive_type: str = "clip",  # "clip" or "dcl"
        temperature: float = 0.07,
        label_smoothing: float = 0.05,    # Balanced
        affinity_weight: float = 0.0,
        affinity_teacher_temp: float = 1.0,
        variance_reg_weight: float = 0.02,  # Light variance regularization
        use_hard_negative_mining: bool = False,
    ):
        super().__init__()
        
        # Main contrastive loss
        if contrastive_type == "clip":
            self.contrastive_loss = CLIPLoss(temperature, label_smoothing)
        elif contrastive_type == "dcl":
            self.contrastive_loss = DecoupledContrastiveLoss(temperature)
        else:
            raise ValueError(f"Unknown contrastive type: {contrastive_type}")
        
        # Affinity mimicking
        self.affinity_weight = affinity_weight
        if affinity_weight > 0:
            self.affinity_loss = AffinityMimickingLoss(
                teacher_temperature=affinity_teacher_temp,
                student_temperature=temperature,
            )
        
        # Variance regularization weight
        self.variance_reg_weight = variance_reg_weight
        
        # Covariance regularization weight (VICReg-style)
        self.cov_reg_weight = variance_reg_weight * 0.5
        
        self.use_hard_negative_mining = use_hard_negative_mining
    
    def forward(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
        brain_emb_unnorm: Optional[torch.Tensor] = None,
        text_emb_unnorm: Optional[torch.Tensor] = None,
        teacher_brain_emb: Optional[torch.Tensor] = None,
        teacher_text_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            brain_emb: Brain embeddings (L2 normalized)
            text_emb: Text embeddings (L2 normalized)
            temperature: Model temperature
            brain_emb_unnorm: Unnormalized brain embeddings for variance reg
            text_emb_unnorm: Unnormalized text embeddings for variance reg
            teacher_brain_emb: Optional teacher brain embeddings
            teacher_text_emb: Optional teacher text embeddings
            
        Returns:
            Loss dictionary
        """
        # Main contrastive loss
        losses = self.contrastive_loss(brain_emb, text_emb, temperature)
        total_loss = losses["loss"]
        
        # Variance regularization to prevent collapse (VICReg-style)
        # Use unnormalized embeddings if provided, otherwise use normalized
        if self.variance_reg_weight > 0:
            brain_for_var = brain_emb_unnorm if brain_emb_unnorm is not None else brain_emb
            text_for_var = text_emb_unnorm if text_emb_unnorm is not None else text_emb
            
            # Variance loss: penalize low variance per dimension
            brain_var = brain_for_var.var(dim=0)
            text_var = text_for_var.var(dim=0)
            
            # Use hinge loss to encourage variance above threshold
            # Target std ~1.0 for embeddings before normalization
            var_loss = torch.mean(F.relu(1.0 - torch.sqrt(brain_var + 1e-4))) + \
                       torch.mean(F.relu(1.0 - torch.sqrt(text_var + 1e-4)))
            
            total_loss = total_loss + self.variance_reg_weight * var_loss
            losses["variance_loss"] = var_loss
            losses["brain_variance"] = brain_var.mean()
            losses["text_variance"] = text_var.mean()
            
            # Covariance loss: decorrelate dimensions (prevents all dims collapsing together)
            if self.cov_reg_weight > 0:
                brain_centered = brain_for_var - brain_for_var.mean(dim=0)
                text_centered = text_for_var - text_for_var.mean(dim=0)
                
                brain_cov = (brain_centered.T @ brain_centered) / (brain_for_var.shape[0] - 1)
                text_cov = (text_centered.T @ text_centered) / (text_for_var.shape[0] - 1)
                
                # Off-diagonal covariance should be zero
                brain_cov_loss = (brain_cov.fill_diagonal_(0) ** 2).sum() / brain_for_var.shape[1]
                text_cov_loss = (text_cov.fill_diagonal_(0) ** 2).sum() / text_for_var.shape[1]
                
                cov_loss = brain_cov_loss + text_cov_loss
                total_loss = total_loss + self.cov_reg_weight * cov_loss
                losses["covariance_loss"] = cov_loss
        
        # Affinity mimicking
        if self.affinity_weight > 0 and teacher_brain_emb is not None:
            affinity_losses = self.affinity_loss(
                brain_emb, text_emb,
                teacher_brain_emb, teacher_text_emb
            )
            total_loss = total_loss + self.affinity_weight * affinity_losses["loss"]
            losses["affinity_loss"] = affinity_losses["loss"]
        
        losses["total_loss"] = total_loss
        
        return losses


def compute_retrieval_metrics(
    brain_emb: torch.Tensor,
    text_emb: torch.Tensor,
    k_values: tuple = (1, 5, 10),
) -> Dict[str, torch.Tensor]:
    """
    Compute retrieval metrics (R@K, MRR).
    
    Args:
        brain_emb: (N, D) brain embeddings
        text_emb: (N, D) text embeddings
        k_values: Tuple of K values for R@K
        
    Returns:
        Dictionary with R@K and MRR for both directions
    """
    batch_size = brain_emb.shape[0]
    device = brain_emb.device
    
    # Similarity matrix
    sim = brain_emb @ text_emb.T  # (N, N)
    
    # Ground truth: diagonal
    labels = torch.arange(batch_size, device=device)
    
    metrics = {}
    
    # Brain-to-text retrieval
    _, indices_b2t = sim.topk(max(k_values), dim=1)
    for k in k_values:
        correct = (indices_b2t[:, :k] == labels.unsqueeze(1)).any(dim=1)
        metrics[f"R@{k}_b2t"] = correct.float().mean()
    
    # MRR for brain-to-text
    ranks_b2t = (sim.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1] + 1
    metrics["MRR_b2t"] = (1.0 / ranks_b2t.float()).mean()
    
    # Text-to-brain retrieval
    _, indices_t2b = sim.T.topk(max(k_values), dim=1)
    for k in k_values:
        correct = (indices_t2b[:, :k] == labels.unsqueeze(1)).any(dim=1)
        metrics[f"R@{k}_t2b"] = correct.float().mean()
    
    # MRR for text-to-brain
    ranks_t2b = (sim.T.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1] + 1
    metrics["MRR_t2b"] = (1.0 / ranks_t2b.float()).mean()
    
    return metrics
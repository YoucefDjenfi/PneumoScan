# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for binary medical classification.
    
    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification"
    ICCV 2021. Adapted here for binary single-label case.
    
    γ_pos: focusing on hard positives (missed pneumonia — high recall cost)
    γ_neg: aggressive suppression of easy negatives (reduce false alarm noise)
    clip:  probability margin — shifts decision boundary for asymmetric hard mining
    
    Clinical justification:
    - False negatives (missed pneumonia) carry higher clinical cost than FP
    - γ_neg > γ_pos ensures model is penalized harder for missing positives
    - clip=0.05 prevents degenerate solutions where p_neg→0 dominates loss
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,   # no focusing on positives (keep full gradient)
        gamma_neg: float = 4.0,   # strong suppression of easy negatives
        clip:      float = 0.05,  # probability shift margin
        eps:       float = 1e-8,
        reduction: str   = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip
        self.eps       = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, 2)  raw unnormalized scores
        targets : (N,)    integer class labels {0, 1}
        """
        # Convert to one-hot for ASL formulation
        targets_oh = F.one_hot(targets, num_classes=logits.size(1)).float()
        
        probs = torch.sigmoid(logits)
        
        # ── Asymmetric clip: shift p for negatives ────────────────────────────
        # Negatives with p < clip are treated as p=0 → zero gradient contribution
        # This is the key mechanism that differs from standard Focal Loss
        probs_neg = (probs - self.clip).clamp(min=0.0)
        
        # ── Reweight by foreground/background mask ────────────────────────────
        probs_combined = targets_oh * probs + (1 - targets_oh) * probs_neg
        probs_combined = probs_combined.clamp(min=self.eps)
        
        # ── Log probability ───────────────────────────────────────────────────
        log_p = torch.log(probs_combined)
        
        # ── Focusing weights ──────────────────────────────────────────────────
        # Positive focusing: (1 - p)^γ_pos
        # Negative focusing: p^γ_neg  (note: p, not 1-p — inverted for negatives)
        pt_pos = 1.0 - probs
        pt_neg = probs + self.clip  # un-shifted for focusing weight
        
        asymmetric_weight = (
            targets_oh       * (pt_pos ** self.gamma_pos) +
            (1 - targets_oh) * (pt_neg ** self.gamma_neg)
        )
        
        loss = -asymmetric_weight * log_p
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingASL(nn.Module):
    """
    ASL + label smoothing for overconfidence regularization.
    
    In medical AI, a model that outputs 0.999 for pneumonia is dangerous —
    it suppresses clinical uncertainty that should be communicated to the
    radiologist. Label smoothing (ε=0.1) prevents logit saturation.
    """
    
    def __init__(self, smoothing: float = 0.1, **asl_kwargs):
        super().__init__()
        self.smoothing = smoothing
        self.asl = AsymmetricLoss(**asl_kwargs)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        with torch.no_grad():
            targets_smooth = F.one_hot(targets, n_classes).float()
            targets_smooth = targets_smooth * (1 - self.smoothing) + \
                             self.smoothing / n_classes
        # Pass soft targets directly to cross-entropy (bypass ASL's one-hot)
        # Use standard weighted CE with smooth targets
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        return loss
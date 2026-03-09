"""
ESM3Di Loss Functions

This module contains all loss functions for training ESM3Di models:
- FocalLoss: Standard focal loss for class imbalance
- CyclicalFocalLoss: Epoch-dependent focal loss with asymmetric weighting
- PLDDTWeightedFocalLoss: Focal loss weighted by pLDDT confidence bins
- PLDDTWeightedCyclicalFocalLoss: Cyclical focal loss with pLDDT weighting
- GammaSchedulerOnPlateau: Scheduler to increase gamma on accuracy plateau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# Default pLDDT bin weights for pLDDT-weighted losses
# Index 0-9 maps to weight for pLDDT bins [0-10), [10-20), ..., [90-100]
DEFAULT_PLDDT_BIN_WEIGHTS = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and focusing on hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma=0, this is equivalent to standard cross-entropy.
    Higher gamma values down-weight easy examples more aggressively.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = None, 
                 reduction: str = 'mean', ignore_index: int = -100):
        """
        Args:
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   Common values: 0.5, 1.0, 2.0, 5.0. Default: 2.0
            alpha: Optional class weight (scalar for binary, tensor for multi-class).
                   If None, no class weighting is applied.
            reduction: Specifies the reduction: 'none', 'mean', 'sum'
            ignore_index: Target value to ignore (default: -100 for masked tokens)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) where N is batch*seq_len, C is num_classes
            targets: Ground truth labels of shape (N,)
        Returns:
            Focal loss value
        """
        # Compute cross-entropy (without reduction)
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        # Get probabilities for the target class
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE) since CE = -log(p_t)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight to loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha is a tensor of per-class weights
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            # Only average over non-ignored elements
            valid_mask = targets != self.ignore_index
            return focal_loss[valid_mask].mean() if valid_mask.any() else focal_loss.sum() * 0.0
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CyclicalFocalLoss(nn.Module):
    """
    Minimal standalone version of lnsmith54/CFL Cyclical_FocalLoss.

    Matches the repo logic closely:
      - single-label classification
      - softmax + log_softmax formulation
      - asymmetric weighting with gamma_pos / gamma_neg
      - hard-class positive weighting with gamma_hc
      - epoch-dependent mixing coefficient eta
      - optional label smoothing

    Args:
        gamma_pos: focal exponent for positive class entries
        gamma_neg: focal exponent for negative class entries
        gamma_hc: exponent for the hard-class positive term
        eps: label smoothing amount
        reduction: "mean", "sum", or "none"
        epochs: total number of epochs used to define eta
        factor: 2 for cyclical, 1 for one-way modified schedule
        ignore_index: Target value to ignore (default: -100)
    """
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        gamma_hc: float = 0.0,
        eps: float = 0.1,
        reduction: str = "mean",
        epochs: int = 200,
        factor: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_hc = gamma_hc
        self.eps = eps
        self.reduction = reduction
        self.epochs = epochs
        self.factor = factor
        self.ignore_index = ignore_index

    def _eta(self, epoch: int) -> float:
        """Compute epoch-dependent mixing coefficient."""
        denom = max(self.epochs - 1, 1)
        if self.factor * epoch < self.epochs:
            eta = 1.0 - self.factor * epoch / denom
        elif self.factor == 1.0:
            eta = 0.0
        else:
            eta = (self.factor * epoch / denom - 1.0) / (self.factor - 1.0)
        return eta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                epoch: int = 0) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
            epoch: current epoch number for eta calculation
        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)
        
        # Handle ignore_index by creating a valid mask
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return inputs.sum() * 0.0
        
        # Filter to valid positions
        valid_inputs = inputs[valid_mask]
        valid_targets = targets[valid_mask]
        
        log_preds = F.log_softmax(valid_inputs, dim=-1)

        # Convert to one-hot targets
        targets_onehot = torch.zeros_like(valid_inputs).scatter_(
            1, valid_targets.long().unsqueeze(1), 1.0
        )
        anti_targets = 1.0 - targets_onehot

        eta = self._eta(epoch)

        # Compute probabilities
        probs = torch.exp(log_preds)
        xs_pos = probs * targets_onehot
        xs_neg = (1.0 - probs) * anti_targets

        asymmetric_w = torch.pow(
            1.0 - xs_pos - xs_neg,
            self.gamma_pos * targets_onehot + self.gamma_neg * anti_targets
        )

        positive_w = torch.pow(
            1.0 + xs_pos,
            self.gamma_hc * targets_onehot
        )

        weights = (1.0 - eta) * asymmetric_w + eta * positive_w
        weighted_log_preds = log_preds * weights

        # Apply label smoothing
        if self.eps > 0:
            smooth_targets = targets_onehot * (1.0 - self.eps) + self.eps / num_classes
        else:
            smooth_targets = targets_onehot

        loss = -(smooth_targets * weighted_log_preds).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            # Return full-size tensor with zeros for ignored positions
            full_loss = torch.zeros(inputs.size(0), device=inputs.device, dtype=loss.dtype)
            full_loss[valid_mask] = loss
            return full_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class PLDDTWeightedFocalLoss(FocalLoss):
    """
    Focal Loss with per-position pLDDT-based weighting using discretized bins.
    
    Expects pLDDT bins (integers 0-9) as input, typically loaded from a
    pLDDT bins FASTA file generated by extract_plddt_bins.py or build_trainingset.py.
    
    Bin mapping:
        0: pLDDT [0, 10)   -> weight from bin_weights[0]
        1: pLDDT [10, 20)  -> weight from bin_weights[1]
        ...
        9: pLDDT [90, 100] -> weight from bin_weights[9]
    
    Higher-confidence positions (higher bins) contribute more to the loss,
    while low-confidence regions are down-weighted or ignored.
    
    The weight_exponent parameter allows sharpening the contrast between
    low and high pLDDT weights (e.g., exponent=2 squares the weights).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = None,
                 reduction: str = 'mean', ignore_index: int = -100,
                 bin_weights: List[float] = None,
                 min_bin: int = 5,
                 weight_exponent: float = 1.0):
        """
        Args:
            gamma: Focal loss focusing parameter. Higher values focus on hard examples.
            alpha: Optional class weight (scalar for binary, tensor for multi-class).
            reduction: 'none', 'mean', or 'sum'
            ignore_index: Target value to ignore (default: -100)
            bin_weights: Weights for each pLDDT bin (0-9). Length must be 10.
                        If None, uses DEFAULT_PLDDT_BIN_WEIGHTS.
            min_bin: Minimum bin to include in loss (default: 5, i.e., pLDDT >= 50).
                    Positions with bin < min_bin get weight 0.
            weight_exponent: Exponent applied to weights to sharpen contrast (default: 1.0).
                            Values > 1 increase contrast (e.g., 2.0 squares weights).
                            Example: weight 0.7 becomes 0.49 with exponent=2.
        """
        # Don't call super().__init__() because we need to set up differently
        nn.Module.__init__(self)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.min_bin = min_bin
        self.weight_exponent = weight_exponent
        
        # Setup bin weights
        if bin_weights is None:
            bin_weights = DEFAULT_PLDDT_BIN_WEIGHTS.copy()
        
        if len(bin_weights) != 10:
            raise ValueError(f"bin_weights must have 10 elements, got {len(bin_weights)}")
        
        # Zero out weights below min_bin
        bin_weights = list(bin_weights)
        for i in range(min_bin):
            bin_weights[i] = 0.0
        
        # Register as buffer so it moves to device with model
        self.register_buffer('bin_weights', torch.tensor(bin_weights, dtype=torch.float32))
    
    def _compute_plddt_weights(self, plddt_bins: torch.Tensor) -> torch.Tensor:
        """
        Convert discretized pLDDT bins to position weights.
        
        Args:
            plddt_bins: Bin indices of shape (N,) with values in [0, 9]
        Returns:
            Weights of shape (N,) in range [0, 1]
        """
        # Clamp bins to valid range
        bins_clamped = torch.clamp(plddt_bins.long(), 0, 9)
        # Index into bin weights
        weights = self.bin_weights[bins_clamped]
        
        # Apply exponent to sharpen contrast between low and high weights
        if self.weight_exponent != 1.0:
            weights = weights ** self.weight_exponent
        
        return weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                plddt_bins: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) where N is batch*seq_len, C is num_classes
            targets: Ground truth labels of shape (N,)
            plddt_bins: Per-position discretized pLDDT bins of shape (N,) with values 0-9.
                       If None, falls back to standard FocalLoss behavior (all weights = 1).
        Returns:
            Weighted focal loss value
        """
        # Compute cross-entropy (without reduction)
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Get probabilities for the target class
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight to loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply pLDDT-based position weighting
        plddt_weights = None
        if plddt_bins is not None:
            plddt_weights = self._compute_plddt_weights(plddt_bins)
            focal_loss = plddt_weights * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            valid_mask = targets != self.ignore_index
            if not valid_mask.any():
                return focal_loss.sum() * 0.0
            
            if plddt_bins is not None:
                # Weighted mean: sum(w * loss) / sum(w) for valid positions
                valid_weights = plddt_weights[valid_mask]
                valid_loss = focal_loss[valid_mask]
                weight_sum = valid_weights.sum()
                if weight_sum > 0:
                    return valid_loss.sum() / weight_sum
                else:
                    return valid_loss.sum() * 0.0
            else:
                return focal_loss[valid_mask].mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PLDDTWeightedCyclicalFocalLoss(nn.Module):
    """
    Cyclical Focal Loss with per-position pLDDT-based weighting.
    
    Combines the epoch-dependent asymmetric weighting of CyclicalFocalLoss
    with pLDDT confidence-based position weighting. This allows the model to:
    
    1. Focus on hard examples via the cyclical focal mechanism
    2. Down-weight low-confidence positions based on pLDDT scores
    
    The pLDDT weighting is applied AFTER the cyclical focal loss computation,
    scaling each position's contribution based on structural confidence.
    
    Bin mapping:
        0: pLDDT [0, 10)   -> weight from bin_weights[0]
        1: pLDDT [10, 20)  -> weight from bin_weights[1]
        ...
        9: pLDDT [90, 100] -> weight from bin_weights[9]
    """
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        gamma_hc: float = 0.0,
        eps: float = 0.1,
        reduction: str = "mean",
        epochs: int = 200,
        factor: float = 2.0,
        ignore_index: int = -100,
        bin_weights: List[float] = None,
        min_bin: int = 5,
        weight_exponent: float = 1.0,
    ):
        """
        Args:
            gamma_pos: focal exponent for positive class entries
            gamma_neg: focal exponent for negative class entries
            gamma_hc: exponent for the hard-class positive term
            eps: label smoothing amount
            reduction: "mean", "sum", or "none"
            epochs: total number of epochs used to define eta
            factor: 2 for cyclical, 1 for one-way modified schedule
            ignore_index: Target value to ignore (default: -100)
            bin_weights: Weights for each pLDDT bin (0-9). Length must be 10.
                        If None, uses DEFAULT_PLDDT_BIN_WEIGHTS.
            min_bin: Minimum bin to include in loss (default: 5, i.e., pLDDT >= 50).
                    Positions with bin < min_bin get weight 0.
            weight_exponent: Exponent applied to weights to sharpen contrast (default: 1.0).
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_hc = gamma_hc
        self.eps = eps
        self.reduction = reduction
        self.epochs = epochs
        self.factor = factor
        self.ignore_index = ignore_index
        self.min_bin = min_bin
        self.weight_exponent = weight_exponent
        
        # Setup bin weights
        if bin_weights is None:
            bin_weights = DEFAULT_PLDDT_BIN_WEIGHTS.copy()
        
        if len(bin_weights) != 10:
            raise ValueError(f"bin_weights must have 10 elements, got {len(bin_weights)}")
        
        # Zero out weights below min_bin
        bin_weights = list(bin_weights)
        for i in range(min_bin):
            bin_weights[i] = 0.0
        
        # Register as buffer so it moves to device with model
        self.register_buffer('bin_weights', torch.tensor(bin_weights, dtype=torch.float32))

    def _eta(self, epoch: int) -> float:
        """Compute epoch-dependent mixing coefficient."""
        denom = max(self.epochs - 1, 1)
        if self.factor * epoch < self.epochs:
            eta = 1.0 - self.factor * epoch / denom
        elif self.factor == 1.0:
            eta = 0.0
        else:
            eta = (self.factor * epoch / denom - 1.0) / (self.factor - 1.0)
        return eta

    def _compute_plddt_weights(self, plddt_bins: torch.Tensor) -> torch.Tensor:
        """
        Convert discretized pLDDT bins to position weights.
        
        Args:
            plddt_bins: Bin indices of shape (N,) with values in [0, 9]
        Returns:
            Weights of shape (N,) in range [0, 1]
        """
        bins_clamped = torch.clamp(plddt_bins.long(), 0, 9)
        weights = self.bin_weights[bins_clamped]
        
        if self.weight_exponent != 1.0:
            weights = weights ** self.weight_exponent
        
        return weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                epoch: int = 0, plddt_bins: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
            epoch: current epoch number for eta calculation
            plddt_bins: Per-position discretized pLDDT bins of shape (N,) with values 0-9.
                       If None, no pLDDT weighting is applied.
        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)
        
        # Handle ignore_index by creating a valid mask
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return inputs.sum() * 0.0
        
        # Filter to valid positions
        valid_inputs = inputs[valid_mask]
        valid_targets = targets[valid_mask]
        valid_plddt = plddt_bins[valid_mask] if plddt_bins is not None else None
        
        log_preds = F.log_softmax(valid_inputs, dim=-1)

        # Convert to one-hot targets
        targets_onehot = torch.zeros_like(valid_inputs).scatter_(
            1, valid_targets.long().unsqueeze(1), 1.0
        )
        anti_targets = 1.0 - targets_onehot

        eta = self._eta(epoch)

        # Compute probabilities
        probs = torch.exp(log_preds)
        xs_pos = probs * targets_onehot
        xs_neg = (1.0 - probs) * anti_targets

        asymmetric_w = torch.pow(
            1.0 - xs_pos - xs_neg,
            self.gamma_pos * targets_onehot + self.gamma_neg * anti_targets
        )

        positive_w = torch.pow(
            1.0 + xs_pos,
            self.gamma_hc * targets_onehot
        )

        weights = (1.0 - eta) * asymmetric_w + eta * positive_w
        weighted_log_preds = log_preds * weights

        # Apply label smoothing
        if self.eps > 0:
            smooth_targets = targets_onehot * (1.0 - self.eps) + self.eps / num_classes
        else:
            smooth_targets = targets_onehot

        # Per-position loss (before pLDDT weighting)
        per_pos_loss = -(smooth_targets * weighted_log_preds).sum(dim=-1)

        # Apply pLDDT weighting
        plddt_weights = None
        if valid_plddt is not None:
            plddt_weights = self._compute_plddt_weights(valid_plddt)
            per_pos_loss = plddt_weights * per_pos_loss

        if self.reduction == "mean":
            if plddt_weights is not None:
                # Weighted mean: sum(w * loss) / sum(w)
                weight_sum = plddt_weights.sum()
                if weight_sum > 0:
                    return per_pos_loss.sum() / weight_sum
                else:
                    return per_pos_loss.sum() * 0.0
            else:
                return per_pos_loss.mean()
        elif self.reduction == "sum":
            return per_pos_loss.sum()
        elif self.reduction == "none":
            # Return full-size tensor with zeros for ignored positions
            full_loss = torch.zeros(inputs.size(0), device=inputs.device, dtype=per_pos_loss.dtype)
            full_loss[valid_mask] = per_pos_loss
            return full_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class GammaSchedulerOnPlateau:
    """
    Scheduler that increases focal loss gamma when accuracy plateaus.
    
    Similar to ReduceLROnPlateau, but increases gamma to focus more on
    hard examples when the model stops improving on accuracy.
    
    Works with FocalLoss, PLDDTWeightedFocalLoss, and any loss with a .gamma attribute.
    """
    def __init__(self, loss_fn, mode: str = 'max', 
                 factor: float = 0.5, patience: int = 2,
                 threshold: float = 1e-3, max_gamma: float = 5.0,
                 verbose: bool = True):
        """
        Args:
            loss_fn: Loss function instance with .gamma attribute to modify
            mode: 'max' for accuracy (want it to increase), 'min' for loss
            factor: Amount to add to gamma when plateau detected
            patience: Number of epochs with no improvement before increasing gamma
            threshold: Threshold for measuring improvement
            max_gamma: Maximum gamma value (prevents over-focusing)
            verbose: Print messages when gamma is updated
        """
        if not hasattr(loss_fn, 'gamma'):
            raise TypeError("loss_fn must have a 'gamma' attribute")
        
        self.loss_fn = loss_fn
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.max_gamma = max_gamma
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = 0
        self._init_is_better()
    
    def _init_is_better(self):
        if self.mode == 'max':
            self.is_better = lambda current, best: current > best + self.threshold
        else:
            self.is_better = lambda current, best: current < best - self.threshold
    
    def step(self, metric: float, epoch: int = None) -> bool:
        """
        Update gamma based on metric value.
        
        Args:
            metric: The metric to monitor (typically accuracy)
            epoch: Optional epoch number for logging
        Returns:
            bool: True if gamma was increased
        """
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        
        current = float(metric)
        
        if self.best is None:
            self.best = current
            return False
        
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            old_gamma = self.loss_fn.gamma
            new_gamma = min(old_gamma + self.factor, self.max_gamma)
            
            if new_gamma > old_gamma:
                self.loss_fn.gamma = new_gamma
                self.num_bad_epochs = 0
                
                if self.verbose:
                    print(f"\n*** Accuracy plateau detected! "
                          f"Increasing gamma: {old_gamma:.2f} -> {new_gamma:.2f} ***")
                return True
        
        return False
    
    @property
    def gamma(self) -> float:
        return self.loss_fn.gamma
    
    def state_dict(self) -> dict:
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'last_epoch': self.last_epoch,
            'gamma': self.loss_fn.gamma,
        }
    
    def load_state_dict(self, state_dict: dict):
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.last_epoch = state_dict['last_epoch']
        self.loss_fn.gamma = state_dict['gamma']

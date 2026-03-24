"""
Iterative Transformer Classification Head with Learned Halting

This module provides an iterative refinement head (PonderNet-style) that can be used
with any encoder model (ESM2, ESM++, T5-based models like ProtT5/Ankh).

The head learns when to stop iterating based on a learned halting probability,
allowing adaptive computation based on sequence complexity.

Based on PonderNet: https://arxiv.org/abs/2107.05407
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from transformers.utils import ModelOutput


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUGate(nn.Module):
    """
    GRU-style gating mechanism for controlled hidden state updates.
    
    This helps with gradient flow during iterative refinement by allowing
    the model to selectively blend previous and current states, similar
    to how GRU cells control information flow.
    
    The mechanism computes:
        - reset gate: r = sigmoid(W_r * [x_new, x_prev])
        - update gate: z = sigmoid(W_z * [x_new, x_prev])
        - candidate: h = tanh(W_h * [x_new, r * x_prev])
        - output: x = z * x_prev + (1 - z) * h
    """
    
    def __init__(self, d_model: int, bias: bool = True):
        """
        Args:
            d_model: Hidden dimension
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        
        # Reset gate: controls how much of previous state to forget
        self.reset_gate = nn.Linear(d_model * 2, d_model, bias=bias)
        
        # Update gate: controls interpolation between old and new
        self.update_gate = nn.Linear(d_model * 2, d_model, bias=bias)
        
        # Candidate transformation
        self.candidate = nn.Linear(d_model * 2, d_model, bias=bias)
        
        # Initialize gates for stable gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable gradient flow."""
        # Initialize biases to zero
        nn.init.zeros_(self.update_gate.bias)
        nn.init.zeros_(self.reset_gate.bias)
        nn.init.zeros_(self.candidate.bias)
        
        # Xavier initialization for weights
        for module in [self.reset_gate, self.update_gate, self.candidate]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x_new: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        Apply GRU-style gating to blend new and previous states.
        
        Args:
            x_new: New state from current iteration (batch, seq, d_model)
            x_prev: Previous state (batch, seq, d_model)
        
        Returns:
            Gated output (batch, seq, d_model)
        """
        # Concatenate inputs for gate computation
        combined = torch.cat([x_new, x_prev], dim=-1)
        
        # Compute gates
        r = torch.sigmoid(self.reset_gate(combined))  # Reset gate
        z = torch.sigmoid(self.update_gate(combined))  # Update gate
        
        # Compute candidate with reset gate applied to previous state
        reset_prev = r * x_prev
        candidate_input = torch.cat([x_new, reset_prev], dim=-1)
        h = torch.tanh(self.candidate(candidate_input))
        
        # Interpolate between previous state and candidate
        # z=1 means keep previous, z=0 means use new candidate
        output = z * x_prev + (1 - z) * h
        
        return output


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but saved with model)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Length of sequence to get positional encoding for
        Returns:
            Positional encoding tensor (1, seq_len, d_model)
        """
        return self.pe[:, :seq_len, :]


class PositionalMixingLayer(nn.Module):
    """
    Learned mixing layer that combines embeddings with positional encoding.
    
    Instead of simply adding positional encoding, this layer concatenates
    the embeddings with positional information and uses a small MLP to
    learn how to mix them together.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        
        # Sinusoidal positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Learned mixing layer: concat(emb, pos_enc) -> d_model
        # Uses a small bottleneck for efficiency
        self.mix_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional information mixed in (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get positional encoding and expand for batch
        pos_enc = self.positional_encoding(seq_len)  # (1, seq_len, d_model)
        pos_enc = pos_enc.expand(batch_size, -1, -1)  # (batch, seq_len, d_model)
        
        # Concatenate embeddings with positional encoding
        combined = torch.cat([x, pos_enc], dim=-1)  # (batch, seq_len, d_model * 2)
        
        # Mix with learned layer + residual connection
        mixed = self.mix_layer(combined)
        
        return self.norm(x + mixed)


@dataclass
class IterativeTransformerOutput(ModelOutput):
    """
    Output class for models with iterative transformer heads.
    Inherits from HuggingFace ModelOutput for DataParallel compatibility.
    
    Attributes:
        loss: Classification loss (if labels provided)
        logits: Weighted combination of all iteration logits (batch, seq, num_labels)
        hidden_states: Encoder hidden states (if requested)
        attentions: Attention weights (if available)
        halt_probs: Per-position halt probabilities (batch, seq, n_iterations)
        halt_reg_loss: KL divergence regularization loss for halting
        n_iterations: Actual number of iterations performed (as 0-dim tensor for DataParallel)
        mean_halt_dist: Mean halt probability distribution across positions
    """
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    halt_probs: Optional[torch.Tensor] = None
    halt_reg_loss: Optional[torch.Tensor] = None
    n_iterations: Optional[torch.Tensor] = None  # 0-dim tensor for DataParallel compatibility
    mean_halt_dist: Optional[torch.Tensor] = None


class IterativeTransformerClassificationHead(nn.Module):
    """
    Iterative Transformer-based classification head with learned halting (PonderNet-style).
    
    Uses self-attention with adaptive iteration - learns when to stop refining predictions.
    The model learns a halting probability at each iteration and combines outputs using
    these probabilities. During inference, can early-stop when confident.
    
    Compatible with any encoder model (ESM2, ESM++, ProtT5, Ankh, etc.)
    
    Args:
        hidden_size: Input dimension from encoder (e.g., 768 for ESM2-650M)
        num_labels: Number of output classes (e.g., 20 for 3Di)
        transformer_dim: Hidden dimension of transformer head (default: 256)
        num_layers: Number of transformer encoder layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        num_heads: Number of attention heads (auto-detected if None)
        use_positional_encoding: Whether to add positional encoding (default: True)
        max_seq_len: Maximum sequence length for positional encoding (default: 8192)
        use_hidden_state_feedback: Feed hidden states back instead of logits (default: True)
        use_gru_gate: Use GRU-style gating for controlled hidden state updates (default: False).
                      Helps with gradient flow by selectively blending previous and current states.
        max_iterations: Maximum number of refinement iterations (default: 5)
        halt_threshold: Cumulative halt probability threshold for early stopping (default: 0.95)
        lambda_p: Weight for halting regularization loss (default: 0.01)
        geometric_prior_p: Parameter for geometric prior (expected iterations ≈ 1/p, default: 0.5)
    
    Example:
        >>> head = IterativeTransformerClassificationHead(
        ...     hidden_size=768,
        ...     num_labels=20,
        ...     max_iterations=5,
        ...     use_gru_gate=True,  # Enable GRU gating for better gradient flow
        ...     geometric_prior_p=0.33  # expects ~3 iterations
        ... )
        >>> # During training
        >>> result = head(encoder_output, attention_mask, return_halt_info=True)
        >>> loss = classification_loss + result['halt_reg_loss']
        >>> 
        >>> # During inference
        >>> head.eval()
        >>> logits = head(encoder_output, attention_mask)  # may early-stop
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        transformer_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
        use_positional_encoding: bool = True,
        max_seq_len: int = 8192,
        use_hidden_state_feedback: bool = True,
        use_gru_gate: bool = False,
        max_iterations: int = 5,
        halt_threshold: float = 0.95,
        lambda_p: float = 0.01,
        geometric_prior_p: float = 0.5,
    ):
        super().__init__()
        
        # Validate inputs
        if transformer_dim <= 0:
            raise ValueError("transformer_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if not 0 < halt_threshold <= 1:
            raise ValueError("halt_threshold must be in (0, 1]")
        if not 0 < geometric_prior_p < 1:
            raise ValueError("geometric_prior_p must be in (0, 1)")
        
        # Auto-detect number of attention heads
        if num_heads is None:
            for h in [8, 4, 2, 1]:
                if transformer_dim % h == 0:
                    num_heads = h
                    break
        
        if num_heads is None or num_heads <= 0 or transformer_dim % num_heads != 0:
            raise ValueError(
                f"Invalid transformer head config: transformer_dim={transformer_dim}, "
                f"num_heads={num_heads}. transformer_dim must be divisible by num_heads."
            )
        
        # Store configuration
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_iterations = max_iterations
        self.halt_threshold = halt_threshold
        self.lambda_p = lambda_p
        self.geometric_prior_p = geometric_prior_p
        self.use_positional_encoding = use_positional_encoding
        self.use_hidden_state_feedback = use_hidden_state_feedback
        self.use_gru_gate = use_gru_gate

        # Input projection from encoder hidden states to transformer dimension
        self.input_projection = (
            nn.Identity()
            if hidden_size == transformer_dim
            else nn.Linear(hidden_size, transformer_dim)
        )
        
        # Positional encoding with learned mixing layer
        if use_positional_encoding:
            self.positional_mixing = PositionalMixingLayer(
                d_model=transformer_dim,
                max_len=max_seq_len,
            )
        else:
            self.positional_mixing = None

        # Feedback projection: map classification logits back to transformer space
        # Only needed if not using hidden state feedback
        if not use_hidden_state_feedback:
            self.feedback_projection = nn.Linear(num_labels, transformer_dim)
        else:
            self.feedback_projection = None

        # GRU gate for controlled hidden state updates between iterations
        # Helps with gradient flow by selectively blending previous and current states
        if use_gru_gate:
            self.gru_gate = GRUGate(transformer_dim)
        else:
            self.gru_gate = None

        # Learnable iteration embedding to distinguish refinement passes
        self.iteration_embedding = nn.Embedding(max_iterations, transformer_dim)

        # Transformer encoder for iterative refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        
        # Classification head
        self.classifier = nn.Linear(transformer_dim, num_labels)

        # Halting network: predicts probability of stopping at each iteration
        self.halt_predictor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, 1),
        )

    def _compute_geometric_prior(self, n_steps: int, device: torch.device) -> torch.Tensor:
        """
        Compute geometric prior probabilities for halting regularization.
        
        The geometric distribution with parameter p has:
        - P(halt at step n) = (1-p)^(n-1) * p
        - Expected number of steps = 1/p
        
        Args:
            n_steps: Number of steps to compute prior for
            device: Target device for tensor
        
        Returns:
            Normalized prior probabilities (n_steps,)
        """
        p = self.geometric_prior_p
        steps = torch.arange(1, n_steps + 1, device=device, dtype=torch.float32)
        prior = p * ((1 - p) ** (steps - 1))
        # Normalize to sum to 1 (truncated geometric)
        prior = prior / prior.sum()
        return prior

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_iterations: bool = False,
        return_halt_info: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with iterative refinement and learned halting.
        
        Args:
            hidden_states: Encoder output (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len) with 1 for valid tokens
            return_all_iterations: If True, include logits from all iterations in output
            return_halt_info: If True, return dict with halt probabilities and regularization loss
        
        Returns:
            If return_halt_info=False:
                logits: Weighted combination (batch_size, seq_len, num_labels)
            If return_halt_info=True:
                dict with:
                    - logits: weighted combination of all iteration logits
                    - halt_probs: (batch, seq, n_iterations) halt probabilities
                    - n_iterations: actual iterations used
                    - halt_reg_loss: KL divergence regularization loss
                    - mean_halt_dist: mean halt distribution for logging
                    - all_logits: list of logits (if return_all_iterations=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Initial projection to transformer dimension
        x = self.input_projection(hidden_states)
        
        # Mix in positional information with learned layer
        if self.positional_mixing is not None:
            x = self.positional_mixing(x)
        
        initial_repr = x

        # Prepare attention mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        # Storage for iteration outputs
        all_logits: List[torch.Tensor] = []
        halt_probs: List[torch.Tensor] = []
        cumulative_halt = torch.zeros(batch_size, seq_len, 1, device=device)
        
        # For early stopping during inference
        halted = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        actual_iterations = 0

        for iteration in range(self.max_iterations):
            actual_iterations = iteration + 1
            
            # Add iteration embedding (broadcast across sequence)
            iter_emb = self.iteration_embedding(
                torch.tensor(iteration, device=device)
            )
            x_iter = x + iter_emb.unsqueeze(0).unsqueeze(0)

            # Transform through encoder with residual connection
            x_iter = self.encoder(x_iter, src_key_padding_mask=src_key_padding_mask)
            x_iter = self.norm(x_iter + initial_repr)

            # Get classification logits for this iteration
            logits = self.classifier(x_iter)
            all_logits.append(logits)

            # Compute halt probability for this iteration
            # λ_n = sigmoid(halt_predictor(state))
            lambda_n = torch.sigmoid(self.halt_predictor(x_iter))  # (batch, seq, 1)
            
            # p_n = λ_n * (1 - cumulative_halt) for n < max_iterations
            # Last iteration gets all remaining probability
            if iteration < self.max_iterations - 1:
                p_n = lambda_n * (1 - cumulative_halt)
            else:
                p_n = 1 - cumulative_halt
            
            halt_probs.append(p_n)
            cumulative_halt = cumulative_halt + p_n

            # Early stopping during inference (not training)
            if not self.training and iteration < self.max_iterations - 1:
                newly_halted = cumulative_halt.squeeze(-1) >= self.halt_threshold
                halted = halted | newly_halted
                if halted.all():
                    break

            # Prepare for next iteration
            if iteration < self.max_iterations - 1:
                if self.use_gru_gate:
                    # GRU-style gating for controlled state updates (best for gradient flow)
                    x = self.gru_gate(x_iter, x)
                elif self.use_hidden_state_feedback:
                    # Feed hidden states directly (preserves more information)
                    x = x_iter
                else:
                    # Feed logits back through projection (original approach)
                    probs = F.softmax(logits, dim=-1)
                    feedback = self.feedback_projection(probs)
                    x = x_iter + feedback

        # Stack halt probabilities: (batch, seq, n_iterations)
        halt_probs_tensor = torch.cat(halt_probs, dim=-1)

        # Compute weighted combination of logits
        # weighted_logits = Σ(p_n * logits_n)
        stacked_logits = torch.stack(all_logits, dim=-1)  # (batch, seq, num_labels, n_iters)
        halt_weights = halt_probs_tensor.unsqueeze(2)  # (batch, seq, 1, n_iters)
        weighted_logits = (stacked_logits * halt_weights).sum(dim=-1)  # (batch, seq, num_labels)

        if return_halt_info:
            # Compute regularization loss (KL divergence with geometric prior)
            with torch.no_grad():
                valid_mask = attention_mask if attention_mask is not None else torch.ones(batch_size, seq_len, device=device)
            
            # Mean halt prob distribution: (n_iterations,)
            mean_halt_dist = (halt_probs_tensor * valid_mask.unsqueeze(-1)).sum(dim=[0, 1])
            mean_halt_dist = mean_halt_dist / (valid_mask.sum() + 1e-8)
            
            # Geometric prior
            prior = self._compute_geometric_prior(actual_iterations, device)
            
            # KL divergence: Σ(p * log(p/q))
            eps = 1e-8
            kl_div = (mean_halt_dist * (torch.log(mean_halt_dist + eps) - torch.log(prior + eps))).sum()
            halt_reg_loss = self.lambda_p * kl_div

            result = {
                'logits': weighted_logits,
                'halt_probs': halt_probs_tensor,
                'n_iterations': actual_iterations,
                'halt_reg_loss': halt_reg_loss,
                'mean_halt_dist': mean_halt_dist.detach(),
            }
            if return_all_iterations:
                result['all_logits'] = all_logits
            return result

        if return_all_iterations:
            return torch.stack(all_logits, dim=0)

        return weighted_logits


class ModelWithIterativeTransformerHead(nn.Module):
    """
    Generic wrapper that adds an iterative transformer head to any encoder model.
    
    Works with:
    - ESM2/ESM++ models (via HuggingFace or custom loaders)
    - T5-based models (ProtT5, Ankh)
    - Any model that returns hidden states
    
    The wrapper extracts hidden states from the base model and passes them to
    the iterative transformer head for classification.
    
    Args:
        base_model: Encoder model with config attribute
        iterative_transformer_head: IterativeTransformerClassificationHead instance
    
    Example:
        >>> base_model = AutoModelForTokenClassification.from_pretrained(...)
        >>> head = IterativeTransformerClassificationHead(
        ...     hidden_size=base_model.config.hidden_size,
        ...     num_labels=20
        ... )
        >>> model = ModelWithIterativeTransformerHead(base_model, head)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        iterative_transformer_head: IterativeTransformerClassificationHead,
    ):
        super().__init__()
        self.base_model = base_model
        self.iterative_transformer_head = iterative_transformer_head

        # Store config for compatibility (required by training loop)
        self.config = base_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> IterativeTransformerOutput:
        """
        Forward pass through encoder + iterative head.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            **kwargs: Additional arguments
                - labels: Optional labels for loss computation
                - return_all_iterations: Return logits from all iterations
                - return_halt_info: Return halt statistics (default: True during training)
        
        Returns:
            IterativeTransformerOutput with logits and optional halt info
        """
        # Extract optional arguments
        labels = kwargs.pop("labels", None)
        return_all_iterations = kwargs.pop("return_all_iterations", False)
        return_halt_info = kwargs.pop("return_halt_info", self.training)

        # Get encoder outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Extract last hidden state (works for both HF models and custom models)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            sequence_output = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            # Fallback: assume outputs is the hidden states directly
            sequence_output = outputs

        # Apply iterative transformer head
        head_output = self.iterative_transformer_head(
            sequence_output,
            attention_mask=attention_mask,
            return_all_iterations=return_all_iterations,
            return_halt_info=return_halt_info,
        )

        # Extract logits and halt info
        if return_halt_info and isinstance(head_output, dict):
            logits = head_output['logits']
            halt_probs = head_output.get('halt_probs')
            halt_reg_loss = head_output.get('halt_reg_loss')
            n_iterations = head_output.get('n_iterations')
            mean_halt_dist = head_output.get('mean_halt_dist')
            # Convert n_iterations to tensor for DataParallel compatibility
            if n_iterations is not None:
                n_iterations = torch.tensor(n_iterations, device=logits.device)
        else:
            logits = head_output
            halt_probs = None
            halt_reg_loss = None
            n_iterations = None
            mean_halt_dist = None

        # Compute classification loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return IterativeTransformerOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            halt_probs=halt_probs,
            halt_reg_loss=halt_reg_loss,
            n_iterations=n_iterations,
            mean_halt_dist=mean_halt_dist,
        )

    # Pass through methods for compatibility
    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)


# Backward compatibility alias
ESMWithIterativeTransformerHead = ModelWithIterativeTransformerHead

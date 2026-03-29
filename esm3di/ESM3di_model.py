
import os
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModel
from .iterative_head import (
    IterativeTransformerClassificationHead,
    IterativeTransformerOutput,
    ModelWithIterativeTransformerHead,
    ESMWithIterativeTransformerHead,  # Backward compat alias
)
from transformers import T5Tokenizer, T5EncoderModel
from transformers.modeling_outputs import TokenClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers.pytorch_utils import Conv1D
from .model_outputs import PLDDT_BIN_VOCAB, TokenClassifierOutputWithPLDDT


# -----------------------------
# Optimizers
# -----------------------------

class Lion(torch.optim.Optimizer):
    """
    Lion optimizer (EvoLved Sign Momentum) - Google Brain 2023.
    
    Uses only the sign of momentum for updates, resulting in:
    - 50% less memory than Adam (only 1 state per parameter)
    - Simpler computation (no sqrt or division)
    - Often faster convergence
    
    Note: Requires ~3-10x lower learning rate than AdamW.
    Recommended: lr=1e-5 to 3e-5, weight_decay=0.1 to 0.3
    """
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), 
                 weight_decay: float = 0.0):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate (use 3-10x lower than AdamW)
            betas: Coefficients for computing running average (beta1, beta2)
                   beta1: for update interpolation (default: 0.9)
                   beta2: for momentum decay (default: 0.99)
            weight_decay: Decoupled weight decay (default: 0.0)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize momentum state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Decoupled weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Lion update: use sign of interpolated momentum
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Update momentum for next iteration
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


# -----------------------------
# Loss Functions (imported from losses.py)
# -----------------------------
from .losses import (
    FocalLoss,
    CyclicalFocalLoss,
    PLDDTWeightedFocalLoss,
    PLDDTWeightedCyclicalFocalLoss,
    GammaSchedulerOnPlateau,
    DEFAULT_PLDDT_BIN_WEIGHTS,
)


# -----------------------------
# CNN Classification Head
# -----------------------------

class CNNClassificationHead(nn.Module):
    """
    Multi-layer CNN classification head for per-token classification.
    Applies 1D convolutions over the sequence dimension.
    """
    def __init__(self, hidden_size: int, num_labels: int, 
                 num_layers: int = 2, kernel_size: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            hidden_size: Input dimension from encoder
            num_labels: Number of output classes
            num_layers: Number of CNN layers (default: 2)
            kernel_size: Convolution kernel size (default: 3)
            dropout: Dropout rate between layers (default: 0.1)
        """
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        for i in range(num_layers):
            in_channels = hidden_size if i == 0 else hidden_size
            out_channels = hidden_size
            
            # Conv1d expects (batch, channels, length)
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            ))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.cnn_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states ):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, seq_len, num_labels)
        """
        # Transpose for Conv1d: (batch, hidden, seq_len)
        x = hidden_states.transpose(1, 2)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Transpose back: (batch, seq_len, hidden)
        x = x.transpose(1, 2)
        
        logits = self.classifier(x)
        return logits


class TransformerClassificationHead(nn.Module):
    """
    Lightweight transformer-encoder classification head for per-token classification.
    Applies a projection -> TransformerEncoder -> linear classifier.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        transformer_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
    ):
        super().__init__()

        if transformer_dim <= 0:
            raise ValueError("transformer_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        if num_heads is None:
            # Prefer larger head counts when divisible; fall back safely.
            for h in [8, 4, 2, 1]:
                if transformer_dim % h == 0:
                    num_heads = h
                    break
        if num_heads is None or num_heads <= 0 or transformer_dim % num_heads != 0:
            raise ValueError(
                f"Invalid transformer head config: transformer_dim={transformer_dim}, num_heads={num_heads}. "
                "transformer_dim must be divisible by num_heads."
            )

        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_projection = (
            nn.Identity()
            if hidden_size == transformer_dim
            else nn.Linear(hidden_size, transformer_dim)
        )

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
        self.classifier = nn.Linear(transformer_dim, num_labels)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) with 1 for valid tokens
        Returns:
            logits: (batch_size, seq_len, num_labels)
        """
        x = self.input_projection(hidden_states)
        residual = x

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x + residual)
        logits = self.classifier(x)
        return logits


class LinearClassificationHead(nn.Module):
    """Simple per-token linear classification head."""

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask=None):
        return self.classifier(hidden_states)


# IterativeTransformerClassificationHead is now imported from iterative_head.py


def _extract_sequence_output(outputs):
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    return outputs.last_hidden_state


class ESMWithLinearHead(nn.Module):
    """Wrapper that uses explicit linear heads for 3Di and optional auxiliary tracks."""

    def __init__(self, base_model, classifier, plddt_head=None, aux_heads=None):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.plddt_head = plddt_head
        self.aux_heads = nn.ModuleDict(aux_heads or {})
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        labels = kwargs.pop("labels", None)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        sequence_output = _extract_sequence_output(outputs)
        logits = self.classifier(sequence_output)
        plddt_logits = self.plddt_head(sequence_output) if self.plddt_head is not None else None
        aux_logits = {name: head(sequence_output) for name, head in self.aux_heads.items()} if self.aux_heads else None

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return TokenClassifierOutputWithPLDDT(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            plddt_logits=plddt_logits,
            aux_logits=aux_logits,
        )


class ESMWithCNNHead(nn.Module):
    """
    Wrapper that replaces the standard classifier with a CNN head.
    Note: Loss is computed externally in training loop to support Focal Loss etc.
    """
    def __init__(self, base_model, cnn_head, plddt_head=None, aux_heads=None):
        super().__init__()
        self.base_model = base_model
        self.cnn_head = cnn_head
        self.plddt_head = plddt_head
        self.aux_heads = nn.ModuleDict(aux_heads or {})
        
        # Store config for compatibility
        self.config = base_model.config
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Extract labels from kwargs if present - don't pass to T5 model
        labels = kwargs.pop("labels", None)
        
        # Get encoder outputs (without classification head)
        # Use output_hidden_states to get the last hidden state
        # T5 does NOT accept labels parameter
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get the last hidden state from the model
        sequence_output = _extract_sequence_output(outputs)
        
        # Apply CNN classification head
        logits = self.cnn_head(sequence_output)
        plddt_logits = self.plddt_head(sequence_output) if self.plddt_head is not None else None
        aux_logits = {name: head(sequence_output) for name, head in self.aux_heads.items()} if self.aux_heads else None
        
        # Note: Loss is computed externally in training loop to support Focal Loss
        # Labels are not passed during training, so loss will be None
        loss = None
        if labels is not None:
            # Fallback for inference or direct usage
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.config.num_labels),
                labels.view(-1)
            )
        
        return TokenClassifierOutputWithPLDDT(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            plddt_logits=plddt_logits,
            aux_logits=aux_logits,
        )
    
    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)


class ESMWithTransformerHead(nn.Module):
    """
    Wrapper that replaces the standard classifier with a small transformer head.
    Note: Loss is computed externally in training loop to support Focal Loss etc.
    """
    def __init__(self, base_model, transformer_head, plddt_head=None, aux_heads=None):
        super().__init__()
        self.base_model = base_model
        self.transformer_head = transformer_head
        self.plddt_head = plddt_head
        self.aux_heads = nn.ModuleDict(aux_heads or {})

        # Store config for compatibility
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Extract labels from kwargs if present - don't pass to T5 model
        labels = kwargs.pop("labels", None)
        
        # Get encoder outputs (without classification head)
        # T5 does NOT accept labels parameter
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        sequence_output = _extract_sequence_output(outputs)

        logits = self.transformer_head(sequence_output, attention_mask=attention_mask)
        plddt_logits = self.plddt_head(sequence_output) if self.plddt_head is not None else None
        aux_logits = {name: head(sequence_output) for name, head in self.aux_heads.items()} if self.aux_heads else None

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return TokenClassifierOutputWithPLDDT(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            plddt_logits=plddt_logits,
            aux_logits=aux_logits,
        )

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)


# IterativeTransformerOutput and ESMWithIterativeTransformerHead are now imported from iterative_head.py


# -----------------------------
# Model Type Detection
# -----------------------------

def is_t5_model(model_name: str) -> bool:
    """
    Detect if a model is a T5-based encoder model (e.g., ProtT5, Ankh).
    These models use T5 architecture but loaded as encoder-only.
    Requires space-separated amino acid tokenization.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        True if model is T5-based encoder, False otherwise
    """
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in ['prot_t5', 'prott5', 'prost_t5', 'prostt5', 'ankh'])


# -----------------------------
# FASTA utilities
# -----------------------------

def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Very simple FASTA parser.
    Returns list of (header_without_>, sequence_string).
    """
    records = []
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip().upper())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records

def discover_lora_target_modules(model) -> List[str]:
    """
    Automatically discover LoRA target modules by finding supported layer types.
    Recursively searches for Linear, Embedding, Conv2d, and Conv1D layers.
    Excludes classifier, head modules, and gated activations (T5DenseGatedActDense)
    which should train fully or not via LoRA.
    """
    # Modules to exclude from LoRA (these train fully or don't work well with LoRA)
    exclude_patterns = [ 'gate']
    include_patterns = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]
    target_modules = set()

    print("Discovering LoRA target modules...")
    def recurse_modules(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Skip modules that shouldn't use LoRA
            if any(pattern in full_name.lower() for pattern in exclude_patterns):
                continue
            
            # Check if this module is a supported type
            if isinstance(child, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                if not any(pattern in full_name.lower() for pattern in exclude_patterns):
                    target_modules.add(full_name)
            
            # Recurse into children
            recurse_modules(child, full_name)
    
    recurse_modules(model)
    target_modules = sorted(list(target_modules))
    return target_modules


def freeze_all_but_lora_and_classifier(model):
    """
    Explicitly freeze everything except:
      - LoRA parameters (names contain 'lora_')
      - classifier head (names contain 'classifier')
      - CNN head (names contain 'cnn_head')
      - Transformer head (names contain 'transformer_head')
      - Iterative transformer head (names contain 'iterative_transformer_head')
    """
    for name, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if (
            "lora_" in name
            or "classifier" in name
            or "plddt_head" in name
            or "aux_heads" in name
            or "cnn_head" in name
            or "transformer_head" in name
            or "iterative_transformer_head" in name
        ):
            p.requires_grad = True


def merge_lora_weights(
    checkpoint_path: str,
    output_path: str = None,
    device: str = None
):
    """
    Merge LoRA weights into base model weights and return a model with
    the original architecture (no LoRA adapters).
    
    This creates a standalone model that can be used without PEFT library.
    
    Args:
        checkpoint_path: Path to checkpoint with LoRA weights
        output_path: Optional path to save merged model. If None, doesn't save.
        device: Device to load model on. Auto-detect if None.
    
    Returns:
        Merged model (standard transformers model without LoRA)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration from checkpoint
    args_dict = checkpoint.get('args', {})
    hf_model_name = args_dict.get('hf_model_name',
                                   'facebook/esm2_t33_650M_UR50D')
    num_labels = len(checkpoint.get('label_vocab', []))
    use_cnn_head = args_dict.get('use_cnn_head', False)
    use_transformer_head = args_dict.get('use_transformer_head', False)
    use_iterative_transformer_head = args_dict.get('use_iterative_transformer_head', False)
    use_plddt_prediction_head = args_dict.get('use_plddt_prediction_head', False)
    plddt_prediction_mode = args_dict.get('plddt_prediction_mode', 'classification')
    lora_r = args_dict.get('lora_r', 8)
    lora_alpha = args_dict.get('lora_alpha', 16)
    lora_dropout = args_dict.get('lora_dropout', 0.05)
    target_modules = checkpoint.get('lora_target_modules', None)
    
    print(f"Base model: {hf_model_name}")
    print(f"Number of labels: {num_labels}")
    print(f"LoRA configuration: r={lora_r}, alpha={lora_alpha}")
    
    # Initialize model with LoRA
    esm_model = ESM3DiModel(
        hf_model_name=hf_model_name,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_cnn_head=use_cnn_head,
        cnn_num_layers=args_dict.get('cnn_num_layers', 2),
        cnn_kernel_size=args_dict.get('cnn_kernel_size', 3),
        cnn_dropout=args_dict.get('cnn_dropout', 0.01),
        use_transformer_head=use_transformer_head,
        use_plddt_prediction_head=use_plddt_prediction_head,
        plddt_num_bins=args_dict.get('plddt_num_bins', len(checkpoint.get('plddt_label_vocab', PLDDT_BIN_VOCAB))),
        plddt_prediction_mode=plddt_prediction_mode,
        transformer_head_dim=args_dict.get('transformer_head_dim', 256),
        transformer_head_layers=args_dict.get('transformer_head_layers', 2),
        transformer_head_dropout=args_dict.get('transformer_head_dropout', 0.1),
        transformer_head_num_heads=args_dict.get('transformer_head_num_heads', None),
        use_iterative_transformer_head=use_iterative_transformer_head,
        iterative_head_max_iterations=args_dict.get('iterative_head_max_iterations', 5),
        iterative_head_halt_threshold=args_dict.get('iterative_head_halt_threshold', 0.95),
        iterative_head_lambda_p=args_dict.get('iterative_head_lambda_p', 0.01),
        iterative_head_prior_p=args_dict.get('iterative_head_prior_p', 0.5),
        use_positional_encoding=args_dict.get('use_positional_encoding', True),
        use_hidden_state_feedback=args_dict.get('use_hidden_state_feedback', True),
    )
    
    # Load LoRA weights (handle potential DataParallel prefix)
    model_with_lora = esm_model.get_model()
    model_state_dict = checkpoint['model_state_dict']
    if any(k.startswith("module.") for k in model_state_dict.keys()):
        model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
    model_with_lora.load_state_dict(model_state_dict)
    model_with_lora = model_with_lora.to(device)
    
    print("✓ LoRA model loaded")
    
    # Merge LoRA weights into base model
    print("Merging LoRA weights into base model...")
    
    # Access the base model depending on whether CNN head is used
    if use_cnn_head:
        # Model structure: ESMWithCNNHead -> PeftModel -> base_model
        peft_model = model_with_lora.base_model
        merged_model = peft_model.merge_and_unload()
        
        # Now we need to recreate the CNN head structure
        # but with the merged base model
        base_esm = merged_model
        cnn_head = model_with_lora.cnn_head
        
        # Create wrapper with merged model
        merged_with_cnn = ESMWithCNNHead(
            base_esm,
            cnn_head,
            plddt_head=getattr(model_with_lora, 'plddt_head', None),
            aux_heads=getattr(model_with_lora, 'aux_heads', None),
        )
        final_model = merged_with_cnn
    elif use_transformer_head:
        # Model structure: ESMWithTransformerHead -> PeftModel -> base_model
        peft_model = model_with_lora.base_model
        merged_model = peft_model.merge_and_unload()

        # Recreate wrapper with merged base model
        base_esm = merged_model
        transformer_head = model_with_lora.transformer_head
        merged_with_transformer = ESMWithTransformerHead(
            base_esm,
            transformer_head,
            plddt_head=getattr(model_with_lora, 'plddt_head', None),
            aux_heads=getattr(model_with_lora, 'aux_heads', None),
        )
        final_model = merged_with_transformer
    elif use_iterative_transformer_head:
        # Model structure: ESMWithIterativeTransformerHead -> PeftModel -> base_model
        peft_model = model_with_lora.base_model
        merged_model = peft_model.merge_and_unload()

        # Recreate wrapper with merged base model
        base_esm = merged_model
        iterative_head = model_with_lora.iterative_transformer_head
        merged_with_iterative = ESMWithIterativeTransformerHead(base_esm, iterative_head)
        final_model = merged_with_iterative
    elif use_plddt_prediction_head and isinstance(model_with_lora, ESMWithLinearHead):
        peft_model = model_with_lora.base_model
        merged_model = peft_model.merge_and_unload()
        final_model = ESMWithLinearHead(
            merged_model,
            model_with_lora.classifier,
            plddt_head=getattr(model_with_lora, 'plddt_head', None),
            aux_heads=getattr(model_with_lora, 'aux_heads', None),
        )
    else:
        # Model structure: PeftModel -> base_model
        final_model = model_with_lora.merge_and_unload()
    
    print("✓ LoRA weights merged")
    
    # Save if output path provided
    if output_path:
        print(f"Saving merged model to: {output_path}")
        
        # Save full merged model state
        save_dict = {
            'model_state_dict': final_model.state_dict(),
            'config': {
                'hf_model_name': hf_model_name,
                'num_labels': num_labels,
                'use_cnn_head': use_cnn_head,
                'use_transformer_head': use_transformer_head,
                'use_iterative_transformer_head': use_iterative_transformer_head,
                'use_plddt_prediction_head': use_plddt_prediction_head,
                'plddt_num_bins': args_dict.get('plddt_num_bins', len(checkpoint.get('plddt_label_vocab', PLDDT_BIN_VOCAB))),
                'plddt_prediction_mode': plddt_prediction_mode,
                'plddt_label_vocab': checkpoint.get('plddt_label_vocab', PLDDT_BIN_VOCAB),
                'transformer_head_dim': args_dict.get('transformer_head_dim', 256),
                'transformer_head_layers': args_dict.get('transformer_head_layers', 2),
                'transformer_head_dropout': args_dict.get('transformer_head_dropout', 0.1),
                'transformer_head_num_heads': args_dict.get('transformer_head_num_heads', None),
                'iterative_head_max_iterations': args_dict.get('iterative_head_max_iterations', 5),
                'iterative_head_halt_threshold': args_dict.get('iterative_head_halt_threshold', 0.95),
                'iterative_head_lambda_p': args_dict.get('iterative_head_lambda_p', 0.01),
                'iterative_head_prior_p': args_dict.get('iterative_head_prior_p', 0.5),
                'use_positional_encoding': args_dict.get('use_positional_encoding', True),
                'use_hidden_state_feedback': args_dict.get('use_hidden_state_feedback', True),
                'label_vocab': checkpoint.get('label_vocab'),
                'mask_label_chars': checkpoint.get('mask_label_chars', ''),
            },
            'training_info': {
                'epoch': checkpoint.get('epoch'),
                'loss': checkpoint.get('loss'),
                'global_step': checkpoint.get('global_step'),
            },
            'merged': True,  # Flag to indicate this is a merged model
        }
        
        torch.save(save_dict, output_path)
        print(f"✓ Merged model saved to {output_path}")
        
        # Calculate size reduction
        original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        merged_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nFile sizes:")
        print(f"  Original (with LoRA): {original_size:.2f} MB")
        print(f"  Merged: {merged_size:.2f} MB")
    
    return final_model


def load_esm3di_from_mlm_checkpoint(
    mlm_checkpoint_path: str,
    hf_model_name: str,
    num_labels: int,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_cnn_head: bool = False,
    cnn_num_layers: int = 2,
    cnn_kernel_size: int = 3,
    cnn_dropout: float = 0.1,
    use_transformer_head: bool = False,
    transformer_head_dim: int = 256,
    transformer_head_layers: int = 2,
    transformer_head_dropout: float = 0.1,
    transformer_head_num_heads: Optional[int] = None,
    use_iterative_transformer_head: bool = False,
    iterative_head_max_iterations: int = 5,
    iterative_head_halt_threshold: float = 0.95,
    iterative_head_lambda_p: float = 0.01,
    iterative_head_prior_p: float = 0.5,
    use_positional_encoding: bool = True,
    use_hidden_state_feedback: bool = True,
    use_gru_gate: bool = False,
    use_plddt_prediction_head: bool = False,
    plddt_num_bins: int = 10,
    plddt_prediction_mode: str = "classification",
    aux_track_num_bins: Optional[dict] = None,
):
    """Load a MaskedLM checkpoint and use its encoder weights for an ESM3DiModel."""
    # Create ESM3DiModel wrapper with LoRA/heads as desired.
    esm3di = ESM3DiModel(
        hf_model_name=hf_model_name,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_cnn_head=use_cnn_head,
        cnn_num_layers=cnn_num_layers,
        cnn_kernel_size=cnn_kernel_size,
        cnn_dropout=cnn_dropout,
        use_transformer_head=use_transformer_head,
        transformer_head_dim=transformer_head_dim,
        transformer_head_layers=transformer_head_layers,
        transformer_head_dropout=transformer_head_dropout,
        transformer_head_num_heads=transformer_head_num_heads,
        use_iterative_transformer_head=use_iterative_transformer_head,
        iterative_head_max_iterations=iterative_head_max_iterations,
        iterative_head_halt_threshold=iterative_head_halt_threshold,
        iterative_head_lambda_p=iterative_head_lambda_p,
        iterative_head_prior_p=iterative_head_prior_p,
        use_positional_encoding=use_positional_encoding,
        use_hidden_state_feedback=use_hidden_state_feedback,
        use_gru_gate=use_gru_gate,
        use_plddt_prediction_head=use_plddt_prediction_head,
        plddt_num_bins=plddt_num_bins,
        plddt_prediction_mode=plddt_prediction_mode,
        aux_track_num_bins=aux_track_num_bins,
    )

    # Load MLM model and extract the encoder backbone.
    # We always instantiate from hf_model_name and then copy state from mlm_checkpoint_path.
    # This avoids config_class mismatch from loading mismatched checkpoint classes directly.
    source_backbone_model = AutoModelForMaskedLM.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
    )

    state_dict = None
    if os.path.isfile(mlm_checkpoint_path) and mlm_checkpoint_path.endswith(".pt"):
        checkpoint = torch.load(mlm_checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    elif os.path.isdir(mlm_checkpoint_path):
        # try loading from HF checkpoint folder
        candidate = None
        for fname in ["pytorch_model.bin", "model.bin"]:
            fpath = os.path.join(mlm_checkpoint_path, fname)
            if os.path.exists(fpath):
                candidate = fpath
                break
        if candidate:
            checkpoint = torch.load(candidate, map_location="cpu")
            # this might be raw state dict or wrapped object
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            # fallback: load direct from path using AutoModelForMaskedLM to handle remote format
            mlm_model = AutoModelForMaskedLM.from_pretrained(
                mlm_checkpoint_path,
                trust_remote_code=True,
            )
            source_backbone = getattr(mlm_model, "base_model", mlm_model)
            state_dict = source_backbone.state_dict()

    if state_dict is not None:
        # Only load matching shapes and ignore classification heads from MLM checkpoint.
        source_sd = source_backbone_model.state_dict()
        filtered_source_sd = {}
        for k, v in state_dict.items():
            if k not in source_sd:
                continue
            if source_sd[k].shape != v.shape:
                continue
            if k.startswith("sequence_head") or k.startswith("lm_head") or k.startswith("classifier"):
                continue
            filtered_source_sd[k] = v
        source_backbone_model.load_state_dict(filtered_source_sd, strict=False)

    source_backbone = getattr(source_backbone_model, "base_model", source_backbone_model)

    # Find the target model for weights assignment
    target_backbone = esm3di.base_model
    if hasattr(esm3di, "model") and hasattr(esm3di.model, "base_model"):
        target_backbone = esm3di.model.base_model

    # Copy only shared weights from source to target, leaving classification head intact.
    source_dict = source_backbone.state_dict()
    target_dict = target_backbone.state_dict()
    filtered = {}
    for k, v in source_dict.items():
        # Skip heads (at least the classification head should stay in target model)
        if k.startswith("sequence_head") or k.startswith("lm_head") or k.startswith("classifier"):
            continue

        # transform key names if necessary (e.g., from `base_model.` prefixes)
        short_k = k
        if short_k.startswith("base_model."):
            short_k = short_k.split("base_model.", 1)[1]
        if short_k.startswith("model."):
            short_k = short_k.split("model.", 1)[1]

        if short_k in target_dict and target_dict[short_k].shape == v.shape:
            filtered[short_k] = v
        # also allow direct key if exact matches already
        elif k in target_dict and target_dict[k].shape == v.shape:
            filtered[k] = v

    load_result = target_backbone.load_state_dict(filtered, strict=False)
    if load_result.missing_keys:
        print(f"[load_esm3di_from_mlm_checkpoint] missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"[load_esm3di_from_mlm_checkpoint] unexpected keys: {len(load_result.unexpected_keys)}")

    return esm3di


class ESM3DiModel:
    """
    Wrapper class for ESM model with LoRA adaptation for 3Di prediction.
    Supports ESM-2, ESM++, and other HuggingFace models.
    """
    def __init__(self, hf_model_name: str, num_labels: int, lora_r: int = 8,
                    lora_alpha: float = 16.0, lora_dropout: float = 0.05,
                    target_modules: List[str] = None,
                    use_cnn_head: bool = False, cnn_num_layers: int = 2,
                    cnn_kernel_size: int = 3, cnn_dropout: float = 0.1,
                    use_transformer_head: bool = False,
                    transformer_head_dim: int = 256,
                    transformer_head_layers: int = 2,
                    transformer_head_dropout: float = 0.1,
                    transformer_head_num_heads: Optional[int] = None,
                    use_iterative_transformer_head: bool = False,
                    iterative_head_max_iterations: int = 5,
                    iterative_head_halt_threshold: float = 0.95,
                    iterative_head_lambda_p: float = 0.01,
                    iterative_head_prior_p: float = 0.5,
                    use_positional_encoding: bool = True,
                    use_hidden_state_feedback: bool = True,
                    use_gru_gate: bool = False,
                    use_plddt_prediction_head: bool = False,
                    plddt_num_bins: int = 10,
                    plddt_prediction_mode: str = "classification",
                    aux_track_num_bins: Optional[dict] = None):
        """
        Initialize ESM model with LoRA configuration.

        Args:
            hf_model_name: HuggingFace model identifier (e.g., 'Synthyra/ESMplusplus_small')
            num_labels: Number of 3Di labels in vocabulary
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names for LoRA. If None, auto-discover.
            use_cnn_head: Whether to use CNN classification head instead of linear
            cnn_num_layers: Number of CNN layers (if use_cnn_head=True)
            cnn_kernel_size: CNN kernel size (if use_cnn_head=True)
            cnn_dropout: Dropout rate for CNN layers (if use_cnn_head=True)
            use_transformer_head: Whether to use a transformer encoder head instead of linear
            transformer_head_dim: Hidden dimension inside transformer head
            transformer_head_layers: Number of transformer encoder layers in head
            transformer_head_dropout: Dropout rate for transformer head
            transformer_head_num_heads: Optional attention head count for transformer head
            use_iterative_transformer_head: Whether to use iterative transformer head with learned halting
            iterative_head_max_iterations: Maximum number of refinement iterations (default: 5)
            iterative_head_halt_threshold: Cumulative halt probability for early stopping (default: 0.95)
            iterative_head_lambda_p: Weight for halting regularization loss (default: 0.01)
            iterative_head_prior_p: Geometric prior parameter for expected iterations (default: 0.5, expects ~2 iterations)
            use_positional_encoding: Add positional encoding mixing to iterative head (default: True)
            use_hidden_state_feedback: Feed hidden states back instead of logits during iteration (default: True)
            use_gru_gate: Use GRU-style gating for controlled hidden state updates (default: False)
        """
        head_count = sum([use_cnn_head, use_transformer_head, use_iterative_transformer_head])
        if head_count > 1:
            raise ValueError("Only one of use_cnn_head, use_transformer_head, or use_iterative_transformer_head can be True")
        if use_plddt_prediction_head and use_iterative_transformer_head:
            raise ValueError("pLDDT prediction head is not yet supported with iterative transformer head")
        if aux_track_num_bins and use_iterative_transformer_head:
            raise ValueError("auxiliary categorical heads are not yet supported with iterative transformer head")
        if plddt_prediction_mode not in {"classification", "regression"}:
            raise ValueError("plddt_prediction_mode must be 'classification' or 'regression'")

        self.hf_model_name = hf_model_name
        self.num_labels = num_labels
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_cnn_head = use_cnn_head
        self.cnn_num_layers = cnn_num_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_dropout = cnn_dropout
        self.use_transformer_head = use_transformer_head
        self.transformer_head_dim = transformer_head_dim
        self.transformer_head_layers = transformer_head_layers
        self.transformer_head_dropout = transformer_head_dropout
        self.transformer_head_num_heads = transformer_head_num_heads
        self.use_iterative_transformer_head = use_iterative_transformer_head
        self.iterative_head_max_iterations = iterative_head_max_iterations
        self.iterative_head_halt_threshold = iterative_head_halt_threshold
        self.iterative_head_lambda_p = iterative_head_lambda_p
        self.iterative_head_prior_p = iterative_head_prior_p
        self.use_positional_encoding = use_positional_encoding
        self.use_hidden_state_feedback = use_hidden_state_feedback
        self.use_gru_gate = use_gru_gate
        self.use_plddt_prediction_head = use_plddt_prediction_head
        self.plddt_num_bins = plddt_num_bins
        self.plddt_prediction_mode = plddt_prediction_mode
        self.aux_track_num_bins = dict(aux_track_num_bins or {})

        # Detect model type - T5 models should use T5ProteinModel instead
        self.is_t5 = is_t5_model(hf_model_name)
        if self.is_t5:
            import warnings
            warnings.warn(
                f"T5-based model detected: {hf_model_name}. "
                "Consider using T5ProteinModel from esm3di.T5Model instead for better support.",
                DeprecationWarning
            )
            print(f"\n⚠️  T5-based model detected: {hf_model_name}")
            print("   For better support, use T5ProteinModel from esm3di.T5Model")

        # Load model using standard HuggingFace approach
        self._load_model()


        # Determine target modules for LoRA
        if target_modules:
            self.target_modules = target_modules
            print(f"\nUsing specified LoRA target modules: {target_modules}")
        else:
            print("\nAuto-discovering LoRA target modules...")
            self.target_modules = discover_lora_target_modules(self.base_model)
            print(f"Discovered target modules: {self.target_modules}")

        # Configure and apply LoRA
        self._setup_lora()

        # Optionally add CNN classification head
        if self.use_cnn_head:
            self._setup_cnn_head()
            print(f"✓ CNN classification head added ({cnn_num_layers} layers)")
        elif self.use_transformer_head:
            self._setup_transformer_head()
            print(
                f"✓ Transformer classification head added "
                f"(dim={self.transformer_head_dim}, layers={self.transformer_head_layers})"
            )
        elif self.use_iterative_transformer_head:
            self._setup_iterative_transformer_head()
            print(
                f"✓ Iterative Transformer head with learned halting "
                f"(dim={self.transformer_head_dim}, layers={self.transformer_head_layers}, "
                f"max_iter={self.iterative_head_max_iterations}, λ_p={self.iterative_head_lambda_p})"
            )
        elif self.use_plddt_prediction_head:
            self._setup_linear_head()
            if self.plddt_prediction_mode == "regression":
                print("✓ Linear classification head added with auxiliary pLDDT regression track")
            else:
                print(f"✓ Linear classification head added with auxiliary pLDDT track ({self.plddt_num_bins} bins)")

        self.freeze_base_model()
        print("✓ LoRA setup complete\n")

        #unfreeze all parameters for inference



    def _load_model(self):
        """
        Load model from HuggingFace using standard AutoModel approach.
        Supports ESM2, ESM++, ProtT5, Ankh, and other transformer models.
        """
        print(f"\nLoading model: {self.hf_model_name}")
        
        if self.is_t5:
            # T5-based model (ProtT5, Ankh, etc.)
            print("  Using T5EncoderModel for encoder-only architecture")
            print("  Space-separated amino acids will be used for tokenization")
            self.base_model = T5EncoderModel.from_pretrained(
                self.hf_model_name,
                trust_remote_code=True
            )
            # T5 models need a classification head added
            # Add a linear layer for token classification
            from torch import nn
            self.base_model.classifier = nn.Linear(
                self.base_model.config.d_model,
                self.num_labels
            )
            # Store config attributes for compatibility
            self.base_model.config.hidden_size = self.base_model.config.d_model
            self.base_model.config.num_labels = self.num_labels
            
            # Load tokenizer - try AutoTokenizer first (works for Ankh), fall back to T5Tokenizer (ProtT5)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_model_name,
                    trust_remote_code=True
                )
            except Exception:
                # ProtT5 may need explicit T5Tokenizer with legacy mode
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.hf_model_name,
                    legacy=True,
                    trust_remote_code=True
                )
            print("✓ Base model loaded (T5EncoderModel)")
            print(f"  Hidden size: {self.base_model.config.d_model}")
            
        else:
            # Standard ESM2/ESM++ loading
            try:
                # Try loading as TokenClassification model directly
                self.base_model = AutoModelForTokenClassification.from_pretrained(
                    self.hf_model_name,
                    num_labels=self.num_labels,
                    trust_remote_code=True,  # Required for ESM++ custom code

                )
                print("✓ Base model loaded (TokenClassification)")
                
            except Exception as e:
                print(f"  Failed to load as TokenClassification: {e}")
                print("  Falling back to AutoModel and adding classification head...")
                # Load base model
                self.base_model = AutoModel.from_pretrained(
                    self.hf_model_name,
                    trust_remote_code=True,  # Required for ESM++ custom code
                )
                self.tokenizer = self.base_model.tokenizer
                print("✓ Base model loaded (AutoModel)")
        

    def _setup_lora(self):
        """Setup LoRA configuration and wrap the base model."""
        # Filter target modules to only include transformer layers (not heads/classifiers)
        # This prevents conflicts with modules_to_save in newer peft versions
        
        safe_target_modules = [
            m for m in self.target_modules 
            if not any(x in m.lower() for x in ['classifier', 'head', 'score', 'pooler', 'lm_head'])
        ]
        
        if not safe_target_modules:
            # Fallback to common transformer layer patterns
            safe_target_modules = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense" ]

            print(f"  Using fallback target modules: {safe_target_modules}")
        
        # Use FEATURE_EXTRACTION for T5 models (they don't support labels in forward())
        # Use TOKEN_CLS for ESM models (standard behavior)
        task_type = TaskType.FEATURE_EXTRACTION if is_t5_model(self.hf_model_name) else TaskType.TOKEN_CLS
        
        lora_config = LoraConfig(
            task_type=task_type,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=safe_target_modules,
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
    
    def _setup_cnn_head(self):
        """Replace standard classifier with CNN classification head."""
        # Get hidden size from base model config
        hidden_size = self.base_model.config.hidden_size
        
        # Create CNN head
        cnn_head = CNNClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            num_layers=self.cnn_num_layers,
            kernel_size=self.cnn_kernel_size,
            dropout=self.cnn_dropout
        )
        
        # Wrap model with CNN head (loss is computed externally during training)
        self.model = ESMWithCNNHead(
            self.model,
            cnn_head,
            plddt_head=self._build_plddt_head(),
            aux_heads=self._build_aux_heads(),
        )

    def _setup_linear_head(self):
        """Replace the default classifier path with explicit linear heads."""
        hidden_size = self.base_model.config.hidden_size
        classifier = LinearClassificationHead(hidden_size=hidden_size, num_labels=self.num_labels)
        self.model = ESMWithLinearHead(
            self.model,
            classifier,
            plddt_head=self._build_plddt_head(),
            aux_heads=self._build_aux_heads(),
        )

    def _setup_transformer_head(self):
        """Replace standard classifier with a small transformer classification head."""
        hidden_size = self.base_model.config.hidden_size

        transformer_head = TransformerClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            transformer_dim=self.transformer_head_dim,
            num_layers=self.transformer_head_layers,
            dropout=self.transformer_head_dropout,
            num_heads=self.transformer_head_num_heads,
        )

        # Wrap model with transformer head (loss is computed externally during training)
        self.model = ESMWithTransformerHead(
            self.model,
            transformer_head,
            plddt_head=self._build_plddt_head(),
            aux_heads=self._build_aux_heads(),
        )

    def _setup_iterative_transformer_head(self):
        """Replace standard classifier with an iterative transformer head with learned halting."""
        hidden_size = self.base_model.config.hidden_size

        iterative_head = IterativeTransformerClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            transformer_dim=self.transformer_head_dim,
            num_layers=self.transformer_head_layers,
            dropout=self.transformer_head_dropout,
            num_heads=self.transformer_head_num_heads,
            use_positional_encoding=self.use_positional_encoding,
            use_hidden_state_feedback=self.use_hidden_state_feedback,
            use_gru_gate=self.use_gru_gate,
            max_iterations=self.iterative_head_max_iterations,
            halt_threshold=self.iterative_head_halt_threshold,
            lambda_p=self.iterative_head_lambda_p,
            geometric_prior_p=self.iterative_head_prior_p,
        )

        # Wrap model with iterative transformer head
        self.model = ESMWithIterativeTransformerHead(self.model, iterative_head)

    def _build_plddt_head(self):
        if not self.use_plddt_prediction_head:
            return None
        hidden_size = self.base_model.config.hidden_size
        plddt_output_dim = 1 if self.plddt_prediction_mode == "regression" else self.plddt_num_bins
        return LinearClassificationHead(hidden_size=hidden_size, num_labels=plddt_output_dim)

    def _build_aux_heads(self):
        hidden_size = self.base_model.config.hidden_size
        heads = {}
        for track_name, n_bins in self.aux_track_num_bins.items():
            if n_bins < 2:
                raise ValueError(f"aux track '{track_name}' must have >=2 bins, got {n_bins}")
            heads[track_name] = LinearClassificationHead(hidden_size=hidden_size, num_labels=n_bins)
        return heads
    
    def freeze_base_model(self):
        """Freeze all base model parameters except LoRA and classifier."""
        freeze_all_but_lora_and_classifier(self.model)

    def get_model(self):
        """Return the LoRA-wrapped model."""
        return self.model
    
    def predict_from_fasta(self,
                          input_fasta_path: str,
                          output_fasta_path: str,
                          model_checkpoint_path: str = None,
                          batch_size: int = 4,
                          device: str = None,
                          output_confidence_fasta: str = None
                          
                          ):
        """
        Predict 3Di sequences from amino acid FASTA file.
        
        Args:
            input_fasta_path: Path to input amino acid FASTA file
            output_fasta_path: Path to save predicted 3Di FASTA file
            model_checkpoint_path: Path to checkpoint to load. If None,
                                   uses current model state.
            batch_size: Batch size for inference
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None.
        
        Returns:
            List of prediction records.
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        
        # Load checkpoint if provided
        if model_checkpoint_path:
            print(f"Loading model from: {model_checkpoint_path}")
            checkpoint = torch.load(model_checkpoint_path,
                                   map_location=device)
            print("✓ Checkpoint loaded")
            print(type(checkpoint))
            print( type(self.model))
            if isinstance(checkpoint, dict) and \
               'model_state_dict' in checkpoint:
                
                # Handle potential DataParallel prefix in checkpoint
                model_state_dict = checkpoint['model_state_dict']
                if any(k.startswith("module.") for k in model_state_dict.keys()):
                    model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
                
                # Use strict=False to handle cases where checkpoint 
                # doesn't include all weights (e.g., frozen embeddings)
                result = self.model.load_state_dict(
                    model_state_dict, 
                    strict=False
                )
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"✓ Loaded model from epoch {epoch}")
                if result.missing_keys:
                    print(f"  (Note: {len(result.missing_keys)} frozen weights "
                          f"not in checkpoint - this is expected)")
            else:
                # Handle potential DataParallel prefix in raw checkpoint
                if any(k.startswith("module.") for k in checkpoint.keys()):
                    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
                result = self.model.load_state_dict(checkpoint, strict=False)
                print("✓ Loaded model checkpoint")
                if result.missing_keys:
                    print(f"  (Note: {len(result.missing_keys)} frozen weights "
                          f"not in checkpoint - this is expected)")

            label_vocab = checkpoint["label_vocab"]
            idx2char = {i: c for i, c in enumerate(label_vocab)}
            args = checkpoint["args"]
            plddt_label_vocab = checkpoint.get("plddt_label_vocab", args.get("plddt_label_vocab", PLDDT_BIN_VOCAB))
            plddt_prediction_mode = args.get("plddt_prediction_mode", "classification")
        else:
            print("No checkpoint provided, using current model state")
            label_vocab = list("ACDEFGHIKLMNPQRSTVWY")[:self.num_labels]
            idx2char = {i: c for i, c in enumerate(label_vocab)}
            plddt_label_vocab = PLDDT_BIN_VOCAB
            plddt_prediction_mode = getattr(self, "plddt_prediction_mode", "classification")

        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize tokenizer
        # Try to get tokenizer from model, fallback to loading from HF
        tokenizer = None
        try:
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'tokenizer'):
                tokenizer = self.model.base_model.tokenizer
        except AttributeError:
            pass
        
        if tokenizer is None:
            if self.is_t5:
                tokenizer = T5Tokenizer.from_pretrained(
                    self.hf_model_name,
                    legacy=True,
                    trust_remote_code=True,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_model_name,
                    do_lower_case=False,
                    trust_remote_code=True,
                )
        
        # Read input FASTA
        print(f"Reading input FASTA: {input_fasta_path}")
        aa_records = read_fasta(input_fasta_path)
        print(f"Found {len(aa_records)} sequences")
        
        # Process sequences in batches
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(aa_records), batch_size):
                batch_records = aa_records[i:i+batch_size]
                headers, raw_seqs = zip(*batch_records)
                
                # T5 models require space-separated amino acids
                if self.is_t5:
                    seqs = [' '.join(list(seq)) for seq in raw_seqs]
                else:
                    seqs = list(raw_seqs)
                
                # Tokenize batch
                enc = tokenizer(
                    list(seqs),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                special_mask = enc["special_tokens_mask"]
                
                # Get predictions
                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                
                logits = outputs.logits
                plddt_logits = getattr(outputs, 'plddt_logits', None)
                
                # Get predicted labels
                pred_labels = torch.argmax(logits, dim=-1)
                pred_plddt_labels = None
                pred_plddt_scores = None
                if plddt_logits is not None:
                    if plddt_prediction_mode == "regression":
                        pred_plddt_scores = torch.clamp(plddt_logits.squeeze(-1), min=0.0, max=100.0)
                    else:
                        pred_plddt_labels = torch.argmax(plddt_logits, dim=-1)
                # Convert predictions to 3Di sequences
                for j, (header, raw_seq) in enumerate(zip(headers, raw_seqs)):
                    pred_3di = []
                    pred_plddt = []
                    pred_plddt_score = []
                    k = 0  # Index into original sequence
                    
                    for pos in range(pred_labels.shape[1]):
                        # Skip special tokens
                        if special_mask[j, pos] == 0:
                            if k < len(raw_seq):
                                label_idx = pred_labels[j, pos].item()
                                pred_char = idx2char.get(label_idx, 'X')
                                pred_3di.append(pred_char)
                                if pred_plddt_scores is not None:
                                    plddt_score = float(pred_plddt_scores[j, pos].item())
                                    pred_plddt_score.append(plddt_score)
                                    plddt_bin = max(0, min(9, int(plddt_score // 10.0)))
                                    pred_plddt.append(str(plddt_bin))
                                elif pred_plddt_labels is not None:
                                    plddt_idx = pred_plddt_labels[j, pos].item()
                                    pred_plddt.append(plddt_label_vocab[plddt_idx] if plddt_idx < len(plddt_label_vocab) else '0')
                                k += 1
                    
                    pred_3di_seq = "".join(pred_3di)
                    pred_plddt_seq = "".join(pred_plddt) if pred_plddt else None
                    
                    # Verify length matches input
                    if len(pred_3di_seq) != len(raw_seq):
                        print(f"Warning: Length mismatch for {header}: "
                              f"AA={len(raw_seq)}, 3Di={len(pred_3di_seq)}")
                    if pred_plddt_seq is not None and len(pred_plddt_seq) != len(raw_seq):
                        print(f"Warning: pLDDT length mismatch for {header}: "
                              f"AA={len(raw_seq)}, pLDDT={len(pred_plddt_seq)}")
                    
                    predictions.append({
                        "header": header,
                        "aa_seq": raw_seq,
                        "pred_3di": pred_3di_seq,
                        "pred_plddt": pred_plddt_seq,
                        "pred_plddt_scores": pred_plddt_score if pred_plddt_score else None,
                    })
                
                if (i + batch_size) % (batch_size * 10) == 0 or \
                   (i + batch_size) >= len(aa_records):
                    processed = min(i + batch_size, len(aa_records))
                    print(f"Processed {processed}/{len(aa_records)} "
                          f"sequences")
        
        # Write output FASTA
        print(f"Writing predictions to: {output_fasta_path}")
        with open(output_fasta_path, 'w') as f:
            for record in predictions:
                f.write(f">{record['header']}\n")
                # Write sequence in lines of 80 characters
                pred_seq = record["pred_3di"]
                for i in range(0, len(pred_seq), 80):
                    f.write(pred_seq[i:i+80] + "\n")

        if output_confidence_fasta:
            plddt_records = [record for record in predictions if record["pred_plddt"] is not None]
            if not plddt_records:
                print("Warning: pLDDT sidecar output requested, but this checkpoint/model has no pLDDT prediction head")
            else:
                print(f"Writing predicted pLDDT bins to: {output_confidence_fasta}")
                with open(output_confidence_fasta, 'w') as f:
                    for record in plddt_records:
                        f.write(f">{record['header']}\n")
                        pred_seq = record["pred_plddt"]
                        for i in range(0, len(pred_seq), 80):
                            f.write(pred_seq[i:i+80] + "\n")
        
        print(f"✓ Prediction complete! {len(predictions)} sequences "
              f"written to {output_fasta_path}")
        return predictions

# -----------------------------
# Dataset
# -----------------------------

class Seq3DiDataset(Dataset):
    """
    Holds (amino_acid_sequence, 3Di_label_sequence, plddt_bins, aux_bins...) tuples.
    Assumes 1:1 correspondence and equal lengths.

    mask_label_chars: characters in 3Di sequences that indicate "masked" positions
                      (e.g. low pLDDT). Only used when pLDDT bins are NOT provided.
                      
                      WITHOUT pLDDT bins (hard masking):
                      - Masked chars are NOT part of label_vocab / model outputs
                      - Those positions are ignored in the loss (labels = -100)
                      
                      WITH pLDDT bins:
                      - mask_label_chars is IGNORED completely
                      - Use the ORIGINAL ground truth 3Di FASTA (no 'X' masking)
                      - pLDDT weighting handles low-confidence positions via soft weighting
    
    plddt_bins_fasta: Optional path to FASTA with pLDDT bins (0-9 per position).
                      If provided, enables pLDDT-weighted loss. Use original 3Di FASTA.

    aux_fastas: Optional dict mapping track name → FASTA path for per-residue auxiliary
                classification targets (e.g. {"bend": "bend_bin.fasta", "torsion": "torsion_bin.fasta"}).
                Characters in each FASTA are converted to int via ord(ch) - ord('0') for digit
                alphabets or via explicit mapping for long alphabets.
    """
    def __init__(
        self,
        aa_fasta: str,
        three_di_fasta: str,
        mask_label_chars: str = "",
        plddt_bins_fasta: str = None,
        aux_fastas: Optional[dict] = None,
    ):
        aa_records = read_fasta(aa_fasta)
        lab_records = read_fasta(three_di_fasta)
        
        # Optionally load pLDDT bins
        self.has_plddt = plddt_bins_fasta is not None
        if self.has_plddt:
            plddt_records = read_fasta(plddt_bins_fasta)
            assert len(plddt_records) == len(aa_records), \
                f"pLDDT bins FASTA has {len(plddt_records)} sequences, expected {len(aa_records)}"
        else:
            plddt_records = None

        # Optionally load aux track FASTAs
        self.aux_track_names: list = list(aux_fastas.keys()) if aux_fastas else []
        aux_records_by_name: dict = {}
        if aux_fastas:
            for track_name, fasta_path in aux_fastas.items():
                recs = read_fasta(fasta_path)
                assert len(recs) == len(aa_records), \
                    f"Aux track '{track_name}' FASTA has {len(recs)} sequences, expected {len(aa_records)}"
                aux_records_by_name[track_name] = recs

        assert len(aa_records) == len(lab_records), "Mismatched number of sequences"

        self.items = []
        all_chars = set()
        
        # When pLDDT is used, ignore mask_label_chars entirely (use original 3Di)
        if self.has_plddt:
            self.mask_label_chars = set()  # Empty - no masking when pLDDT active
        else:
            self.mask_label_chars = set(mask_label_chars) if mask_label_chars else set()

        for idx, ((h_aa, seq_aa), (h_lab, seq_lab)) in enumerate(zip(aa_records, lab_records)):
            if len(seq_aa) != len(seq_lab):
                raise ValueError(
                    f"Length mismatch {h_aa}/{h_lab}: {len(seq_aa)} vs {len(seq_lab)}"
                )
            
            # Get pLDDT bins if available
            if self.has_plddt:
                h_plddt, plddt_seq = plddt_records[idx]
                if len(plddt_seq) != len(seq_aa):
                    raise ValueError(
                        f"pLDDT length mismatch {h_aa}: AA={len(seq_aa)}, pLDDT={len(plddt_seq)}"
                    )
            else:
                plddt_seq = None

            # Get aux track sequences
            aux_seqs: dict = {}
            for track_name in self.aux_track_names:
                h_aux, aux_seq = aux_records_by_name[track_name][idx]
                if len(aux_seq) != len(seq_aa):
                    raise ValueError(
                        f"Aux track '{track_name}' length mismatch {h_aa}: "
                        f"AA={len(seq_aa)}, track={len(aux_seq)}"
                    )
                aux_seqs[track_name] = aux_seq

            self.items.append((h_aa, seq_aa, seq_lab, plddt_seq, aux_seqs))
            all_chars.update(seq_lab)

        # Build vocab: exclude masked characters (when not using pLDDT)
        label_chars = sorted(ch for ch in all_chars if ch not in self.mask_label_chars)
        self.label_vocab = label_chars
        self.char2idx = {c: i for i, c in enumerate(self.label_vocab)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # (header, aa_seq, 3di_seq, plddt_bins_str_or_None, aux_seqs_dict)
        return self.items[idx]

# -----------------------------
# Collate with HF tokenizer
# -----------------------------
def make_collate_fn(
    tokenizer,
    char2idx,
    mask_label_chars: str = "",
    include_plddt: bool = False,
    is_t5: bool = False,
    max_seq_length: int = None,
    aux_track_names: Optional[list] = None,
):
    """
    - Tokenizes AA sequences with HF tokenizer (ESM-2, ESM++, ProtT5, etc.).
    - Uses special_tokens_mask to align per-residue 3Di labels
      to non-special tokens.
    - Optionally includes pLDDT bins for weighted loss.
    - Optionally includes per-residue auxiliary track bins (bend, torsion, …).
    - Supports T5 models with space-separated amino acid tokenization.
    
    Masking behavior:
        WITHOUT pLDDT (include_plddt=False):
            - Positions with chars in mask_label_chars are set to -100 (ignored in loss)
        
        WITH pLDDT (include_plddt=True):
            - mask_label_chars is COMPLETELY IGNORED
            - Use the ORIGINAL ground truth 3Di FASTA (no 'X' characters)
            - pLDDT bins handle low-confidence positions via soft weighting
    
    Args:
        tokenizer: HuggingFace tokenizer
        char2idx: Mapping from 3Di characters to label indices
        mask_label_chars: Characters to ignore in loss (ONLY when NOT using pLDDT)
        include_plddt: Whether to include pLDDT bins (disables masking)
        is_t5: Whether using T5-based model (requires space-separated AA)
        max_seq_length: Maximum sequence length (truncates longer sequences).
        aux_track_names: List of auxiliary track names to include from batch items
                         (must match Seq3DiDataset.aux_track_names).
    """
    mask_set = set(mask_label_chars) if mask_label_chars else set()
    _aux_track_names = list(aux_track_names) if aux_track_names else []

    # Build per-character → bin index mapping for BIN_ALPHABET_BASE characters.
    # Characters '0'-'9' map to 0-9, 'A'-'Z' to 10-35, 'a'-'z' to 36-61.
    _BIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _char_to_bin = {ch: i for i, ch in enumerate(_BIN_ALPHABET)}

    def _aux_char_to_bin(ch: str) -> int:
        """Convert a single aux-track character to its bin index."""
        if ch in _char_to_bin:
            return _char_to_bin[ch]
        raise ValueError(
            f"Aux track character '{ch}' not in BIN_ALPHABET. "
            "Ensure tracks.py produced valid tokens."
        )

    def collate(batch):
        # Batch items are (header, aa_seq, 3di_seq, plddt_seq_or_None, aux_seqs_dict)
        if len(batch[0]) == 5:
            headers, aa_seqs, label_seqs, plddt_seqs, aux_seqs_list = zip(*batch)
        elif len(batch[0]) == 4:
            headers, aa_seqs, label_seqs, plddt_seqs = zip(*batch)
            aux_seqs_list = [{}] * len(headers)
        else:
            headers, aa_seqs, label_seqs = zip(*batch)
            plddt_seqs = [None] * len(headers)
            aux_seqs_list = [{}] * len(headers)

        # T5 models require space-separated amino acids
        if is_t5:
            aa_seqs = [' '.join(list(seq)) for seq in aa_seqs]

        # Standard HuggingFace tokenization
        enc = tokenizer(
            list(aa_seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        input_ids = enc["input_ids"]              # [B, T]
        attention_mask = enc["attention_mask"]    # [B, T]
        special_mask = enc["special_tokens_mask"] # [B, T]

        batch_size, max_len = input_ids.shape
        labels = torch.full(
            (batch_size, max_len),
            -100,  # ignore_index for CE
            dtype=torch.long,
        )
        
        # Initialize pLDDT bins tensor if needed
        has_plddt = include_plddt and plddt_seqs[0] is not None
        if has_plddt:
            plddt_bins = torch.zeros(
                (batch_size, max_len),
                dtype=torch.long,
            )
        else:
            plddt_bins = None

        # Initialize aux track bin tensors (-100 = ignore_index for CE)
        has_aux = bool(_aux_track_names) and bool(aux_seqs_list[0])
        aux_bins: dict = {}
        if has_aux:
            for track_name in _aux_track_names:
                aux_bins[track_name] = torch.full(
                    (batch_size, max_len), -100, dtype=torch.long
                )

        for i, lab_seq in enumerate(label_seqs):
            plddt_seq = plddt_seqs[i] if has_plddt else None
            aux_seqs_i: dict = aux_seqs_list[i] if has_aux else {}
            k = 0  # index into residue sequences
            for j in range(max_len):
                if special_mask[i, j] == 1:
                    # CLS/EOS/PAD etc.
                    labels[i, j] = -100
                    if has_plddt:
                        plddt_bins[i, j] = 0  # zero weight for special tokens
                    # aux stays -100 (ignore)
                else:
                    if k < len(lab_seq):
                        ch = lab_seq[k]
                        # When pLDDT weighting is used, no masking - use original 3Di directly
                        # pLDDT bins handle low-confidence positions via soft weighting
                        if ch in mask_set and not has_plddt:
                            # masked -> ignore in loss (only when NOT using pLDDT)
                            labels[i, j] = -100
                        else:
                            # Get label index - char must be in vocab
                            if ch in char2idx:
                                labels[i, j] = char2idx[ch]
                            else:
                                raise ValueError(
                                    f"Label char '{ch}' not in vocabulary. "
                                    f"When using pLDDT weighting, use the original 3Di FASTA."
                                )
                        
                        # Set pLDDT bin for this position
                        if has_plddt and plddt_seq is not None:
                            plddt_bins[i, j] = int(plddt_seq[k])

                        # Set aux track bins for this position
                        if has_aux:
                            for track_name in _aux_track_names:
                                track_seq = aux_seqs_i.get(track_name, "")
                                if k < len(track_seq):
                                    aux_bins[track_name][i, j] = _aux_char_to_bin(track_seq[k])
                                # else stays -100
                        
                        k += 1
                    else:
                        labels[i, j] = -100
                        if has_plddt:
                            plddt_bins[i, j] = 0
                        # aux stays -100
            
            # Note: k may be < len(lab_seq) if sequence was truncated
            # This is expected behavior when max_seq_length is set

        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if has_plddt:
            batch_out["plddt_bins"] = plddt_bins

        if has_aux:
            batch_out["aux_bins"] = aux_bins
        
        return batch_out

    return collate

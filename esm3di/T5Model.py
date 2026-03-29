"""
T5-based Protein Language Model Support for ESM3Di

This module provides support for T5-based protein language models including:
- Rostlab/ProstT5: Bi-directional translation between amino acid and 3Di sequences
- ElnaggarLab/Ankh: T5-based protein encoder models
- Rostlab/prot_t5_xl_half_bfloat16: ProtT5-XL model

These models use T5 architecture but are loaded as encoder-only models.
They require space-separated amino acid tokenization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

from transformers import (
    T5Tokenizer, 
    T5EncoderModel, 
    AutoTokenizer,
    AutoConfig,
)
from .ESM3di_model import read_fasta
from transformers.modeling_outputs import TokenClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType
from .model_outputs import TokenClassifierOutputWithPLDDT

# Import iterative head from shared module
from .iterative_head import (
    IterativeTransformerClassificationHead,
    IterativeTransformerOutput,
    ModelWithIterativeTransformerHead,
)


# -----------------------------
# Model Type Detection
# -----------------------------

def is_t5_model(model_name: str) -> bool:
    """
    Detect if a model is a T5-based encoder model.
    These models use T5 architecture but are loaded as encoder-only.
    Requires space-separated amino acid tokenization.
    
    Supported models:
    - Rostlab/ProstT5 (bi-directional AA <-> 3Di)
    - Rostlab/prot_t5_* (ProtT5 variants)
    - ElnaggarLab/ankh* (Ankh models)
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        True if model is T5-based encoder, False otherwise
    """
    model_lower = model_name.lower()
    t5_patterns = [
        'prost_t5', 'prostt5',  # ProstT5
        'prot_t5', 'prott5',     # ProtT5
        'ankh',                  # Ankh models
    ]
    return any(pattern in model_lower for pattern in t5_patterns)


def is_prostt5_model(model_name: str) -> bool:
    """
    Detect if a model is specifically ProstT5 (Rostlab/ProstT5).
    ProstT5 is bi-directional and can translate between AA and 3Di.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        True if model is ProstT5, False otherwise
    """
    model_lower = model_name.lower()
    return 'prost' in model_lower and 't5' in model_lower


# -----------------------------
# Classification Heads for T5
# -----------------------------

class CNNClassificationHead(nn.Module):
    """
    CNN-based classification head for per-residue prediction.
    Uses 1D convolutions to capture local sequence context.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        layers = []
        for i in range(num_layers):
            in_channels = hidden_size if i == 0 else hidden_size
            layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.conv_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        
        Returns:
            logits: (batch, seq_len, num_labels)
        """
        # (batch, seq_len, hidden) -> (batch, hidden, seq_len)
        x = hidden_states.transpose(1, 2)
        x = self.conv_layers(x)
        # (batch, hidden, seq_len) -> (batch, seq_len, hidden)
        x = x.transpose(1, 2)
        logits = self.classifier(x)
        return logits


class TransformerClassificationHead(nn.Module):
    """
    Transformer-based classification head for per-residue prediction.
    Uses self-attention to capture global sequence context.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        transformer_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size, transformer_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        self.classifier = nn.Linear(transformer_dim, num_labels)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) optional padding mask
        
        Returns:
            logits: (batch, seq_len, num_labels)
        """
        x = self.projection(hidden_states)
        residual = x
        
        src_key_padding_mask = None
        if attention_mask is not None:
            # Convert attention mask to padding mask (1 = masked, 0 = valid)
            src_key_padding_mask = (attention_mask == 0)
        
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x + residual)
        logits = self.classifier(x)
        return logits


class LinearClassificationHead(nn.Module):
    """Simple per-token linear classification head."""

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.linear(hidden_states)


# IterativeTransformerClassificationHead is now imported from iterative_head.py
# It includes learned halting (PonderNet-style) for adaptive iteration


# -----------------------------
# T5 Model Wrapper
# -----------------------------

class T5WithClassificationHead(nn.Module):
    """
    Wrapper that combines T5EncoderModel with a classification head.
    Returns TokenClassifierOutput for compatibility with training loop.
    Supports IterativeTransformerClassificationHead with halt info.
    """
    def __init__(
        self,
        base_model: nn.Module,
        classification_head: nn.Module,
        num_labels: int,
        plddt_head: Optional[nn.Module] = None,
        aux_heads: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.classification_head = classification_head
        self.num_labels = num_labels
        self.plddt_head = plddt_head
        self.aux_heads = nn.ModuleDict(aux_heads or {})
        
        # Store config for compatibility with training loop
        self.config = base_model.config
        self.config.num_labels = num_labels
        
        # Check if using iterative head
        self._uses_iterative_head = isinstance(
            classification_head, IterativeTransformerClassificationHead
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass compatible with ESM3Di training loop.
        
        Args:
            input_ids: (batch, seq_len) tokenized input
            attention_mask: (batch, seq_len) attention mask
            **kwargs: Additional arguments
                - labels: Optional labels for loss computation
                - return_halt_info: Return halt statistics (iterative head only)
                - return_all_iterations: Return all iteration outputs
        
        Returns:
            TokenClassifierOutput or IterativeTransformerOutput
        """
        # Remove labels from kwargs - loss computed externally
        labels = kwargs.pop("labels", None)
        return_halt_info = kwargs.pop("return_halt_info", self.training and self._uses_iterative_head)
        return_all_iterations = kwargs.pop("return_all_iterations", False)
        
        # Get encoder outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last hidden state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            sequence_output = outputs.hidden_states[-1]
        else:
            sequence_output = outputs.last_hidden_state
        
        # Apply classification head
        if self._uses_iterative_head:
            head_output = self.classification_head(
                sequence_output, 
                attention_mask=attention_mask,
                return_halt_info=return_halt_info,
                return_all_iterations=return_all_iterations,
            )
            
            if return_halt_info and isinstance(head_output, dict):
                logits = head_output['logits']
                halt_probs = head_output.get('halt_probs')
                halt_reg_loss = head_output.get('halt_reg_loss')
                n_iterations = head_output.get('n_iterations')
                mean_halt_dist = head_output.get('mean_halt_dist')
            else:
                logits = head_output
                halt_probs = None
                halt_reg_loss = None
                n_iterations = None
                mean_halt_dist = None
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
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
        
        # Non-iterative heads
        if isinstance(self.classification_head, TransformerClassificationHead):
            logits = self.classification_head(sequence_output, attention_mask)
        else:
            logits = self.classification_head(sequence_output)
        plddt_logits = self.plddt_head(sequence_output) if self.plddt_head is not None else None
        aux_logits = {name: head(sequence_output) for name, head in self.aux_heads.items()} if self.aux_heads else None
        
        # Compute loss if labels provided (for inference/eval)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
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


# -----------------------------
# LoRA Module Discovery
# -----------------------------

def discover_t5_lora_modules(model: nn.Module) -> List[str]:
    """
    Discover suitable LoRA target modules in a T5 model.
    Targets attention and feed-forward layers while excluding classifiers.
    
    Args:
        model: T5 model to analyze
    
    Returns:
        List of module names suitable for LoRA
    """
    target_modules = []
    exclude_patterns = ['classifier', 'head', 'score', 'pooler', 'lm_head', 'shared']
    
    for name, module in model.named_modules():
        # Target Linear layers in attention and feed-forward
        if isinstance(module, nn.Linear):
            # Skip excluded patterns
            if any(excl in name.lower() for excl in exclude_patterns):
                continue
            
            # T5 attention patterns: q, k, v, o projections
            # T5 FFN patterns: wi_0, wi_1, wo (gated), wi, wo (standard)
            target_patterns = ['.q', '.k', '.v', '.o', 'wi', 'wo']
            if any(pat in name for pat in target_patterns):
                target_modules.append(name)
    
    return target_modules


# -----------------------------
# Main T5 Model Class
# -----------------------------

class T5ProteinModel:
    """
    T5-based protein language model with LoRA fine-tuning support.
    
    Supports:
    - Rostlab/ProstT5: Bi-directional AA <-> 3Di translation
    - Rostlab/prot_t5_*: ProtT5 variants
    - ElnaggarLab/ankh*: Ankh models
    
    Features:
    - T5EncoderModel for encoder-only usage
    - Space-separated amino acid tokenization
    - CNN or Transformer classification heads
    - LoRA fine-tuning with FEATURE_EXTRACTION task type
    
    Usage:
        model = T5ProteinModel(
            hf_model_name="Rostlab/ProstT5",
            num_labels=20,
            use_cnn_head=True
        )
        wrapped_model = model.get_model()
    """
    
    def __init__(
        self,
        hf_model_name: str,
        num_labels: int,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        use_cnn_head: bool = False,
        cnn_num_layers: int = 2,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.1,
        use_transformer_head: bool = False,
        transformer_head_dim: int = 256,
        transformer_head_layers: int = 2,
        transformer_head_dropout: float = 0.1,
        transformer_head_num_heads: int = 4,
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
        aux_track_num_bins: Optional[Dict[str, int]] = None,
        half_precision: bool = False,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize T5 protein model with LoRA.
        
        Args:
            hf_model_name: HuggingFace model identifier
            num_labels: Number of output labels (3Di vocabulary size)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout rate
            target_modules: Specific modules to apply LoRA (auto-discover if None)
            use_cnn_head: Use CNN classification head
            cnn_num_layers: Number of CNN layers
            cnn_kernel_size: CNN kernel size
            cnn_dropout: CNN dropout rate
            use_transformer_head: Use Transformer classification head
            transformer_head_dim: Transformer head hidden dimension
            transformer_head_layers: Number of Transformer layers
            transformer_head_dropout: Transformer dropout rate
            transformer_head_num_heads: Number of attention heads
            use_iterative_transformer_head: Use iterative Transformer head with learned halting
            iterative_head_max_iterations: Maximum iterations for iterative head
            iterative_head_halt_threshold: Halt probability threshold for early stopping
            iterative_head_lambda_p: Weight for halt regularization loss
            iterative_head_prior_p: Geometric prior parameter (expected iterations ≈ 1/p)
            use_positional_encoding: Add positional encoding mixing to iterative head
            use_hidden_state_feedback: Feed hidden states back instead of logits during iteration
            use_gru_gate: Use GRU-style gating for controlled hidden state updates (helps with gradient flow)
            use_plddt_prediction_head: Add optional auxiliary per-token pLDDT bin head
            plddt_num_bins: Number of pLDDT bins to predict
            plddt_prediction_mode: Auxiliary pLDDT prediction mode ('classification' or 'regression')
            aux_track_num_bins: Optional mapping of extra categorical track names to bin counts
            half_precision: Load model in half precision (bfloat16/float16)
            gradient_checkpointing: Enable gradient checkpointing to save memory (slower training)
        """
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
        self.half_precision = half_precision
        self.gradient_checkpointing = gradient_checkpointing
        
        self.is_prostt5 = is_prostt5_model(hf_model_name)
        
        print(f"\n⚡ Loading T5-based protein model: {hf_model_name}")
        if self.is_prostt5:
            print("   Model type: ProstT5 (bi-directional AA <-> 3Di)")
        print("   Using space-separated amino acid tokenization")
        
        # Load model and tokenizer
        self._load_model()
        
        # Discover or use provided target modules
        if target_modules:
            self.target_modules = target_modules
            print(f"\nUsing specified LoRA target modules: {target_modules}")
        else:
            print("\nAuto-discovering LoRA target modules...")
            self.target_modules = discover_t5_lora_modules(self.base_model)
            print(f"Discovered {len(self.target_modules)} target modules")
        
        # Setup LoRA
        self._setup_lora()
        
        # Setup classification head
        if self.use_cnn_head:
            self._setup_cnn_head()
            print(f"✓ CNN classification head added ({cnn_num_layers} layers)")
        elif self.use_iterative_transformer_head:
            self._setup_iterative_transformer_head()
            print(f"✓ Iterative Transformer classification head added "
                  f"(dim={transformer_head_dim}, layers={transformer_head_layers}, "
                  f"max_iter={iterative_head_max_iterations}, pos_enc={use_positional_encoding})")
        elif self.use_transformer_head:
            self._setup_transformer_head()
            print(f"✓ Transformer classification head added "
                  f"(dim={transformer_head_dim}, layers={transformer_head_layers})")
        else:
            # Default: simple linear classifier
            self._setup_linear_head()
            print("✓ Linear classification head added")
        
        # Freeze base model except LoRA
        self._freeze_base_model()
        print("✓ LoRA setup complete\n")
    
    def _load_model(self):
        """Load T5EncoderModel and tokenizer."""
        print(f"\nLoading T5EncoderModel: {self.hf_model_name}")
        
        # Determine dtype
        dtype = None
        if self.half_precision:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"  Using half precision: {dtype}")
        
        # Load config first to check settings
        config = AutoConfig.from_pretrained(
            self.hf_model_name,
            trust_remote_code=True
        )
        
        # Load encoder model
        self.base_model = T5EncoderModel.from_pretrained(
            self.hf_model_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        # Store hidden size for classification head
        self.hidden_size = self.base_model.config.d_model
        
        # Ensure config has required attributes
        self.base_model.config.hidden_size = self.hidden_size
        self.base_model.config.num_labels = self.num_labels
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Resize token embeddings if tokenizer vocab is larger than model embeddings
        # This is important when using custom tokenizers or extended vocabularies
        embedding_size = self.base_model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            print(f"  Resizing token embeddings: {embedding_size} -> {len(self.tokenizer)}")
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            print("  Gradient checkpointing: ENABLED (saves ~30-50% memory, slower training)")
        
        print(f"✓ Model loaded (hidden_size={self.hidden_size})")
    
    def _load_tokenizer(self):
        """Load appropriate tokenizer for the model."""
        try:
            # Try AutoTokenizer first (works for most models)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                trust_remote_code=True
            )
            print("  Tokenizer: AutoTokenizer")
        except Exception:
            try:
                # Fall back to T5Tokenizer with legacy mode
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.hf_model_name,
                    legacy=True,
                    trust_remote_code=True
                )
                print("  Tokenizer: T5Tokenizer (legacy)")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for {self.hf_model_name}: {e}")
    
    def _setup_lora(self):
        """Setup LoRA configuration."""
        # Filter out classifier/head modules
        safe_modules = [
            m for m in self.target_modules
            if not any(x in m.lower() for x in ['classifier', 'head', 'score', 'lm_head'])
        ]
        
        if not safe_modules:
            # Fallback to common T5 attention patterns
            safe_modules = ['q', 'k', 'v', 'o', 'wi', 'wo']
            print(f"  Using fallback target modules: {safe_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # T5 doesn't support labels in forward
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=safe_modules,
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
    
    def _setup_cnn_head(self):
        """Setup CNN classification head."""
        cnn_head = CNNClassificationHead(
            hidden_size=self.hidden_size,
            num_labels=self.num_labels,
            num_layers=self.cnn_num_layers,
            kernel_size=self.cnn_kernel_size,
            dropout=self.cnn_dropout
        )
        
        self.model = T5WithClassificationHead(
            self.model,
            cnn_head,
            self.num_labels,
            plddt_head=self._build_plddt_head(),
            aux_heads=self._build_aux_heads(),
        )
    
    def _setup_transformer_head(self):
        """Setup Transformer classification head."""
        transformer_head = TransformerClassificationHead(
            hidden_size=self.hidden_size,
            num_labels=self.num_labels,
            transformer_dim=self.transformer_head_dim,
            num_layers=self.transformer_head_layers,
            dropout=self.transformer_head_dropout,
            num_heads=self.transformer_head_num_heads
        )
        
        self.model = T5WithClassificationHead(
            self.model,
            transformer_head,
            self.num_labels,
            plddt_head=self._build_plddt_head(),
            aux_heads=self._build_aux_heads(),
        )
    
    def _setup_iterative_transformer_head(self):
        """Setup Iterative Transformer classification head with learned halting."""
        iterative_head = IterativeTransformerClassificationHead(
            hidden_size=self.hidden_size,
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
        
        self.model = T5WithClassificationHead(
            self.model,
            iterative_head,
            self.num_labels
        )
    
    def _setup_linear_head(self):
        """Setup simple linear classification head."""
        self.model = T5WithClassificationHead(
            self.model,
            LinearClassificationHead(self.hidden_size, self.num_labels),
            self.num_labels,
            plddt_head=self._build_plddt_head(),
            aux_heads=self._build_aux_heads(),
        )

    def _build_plddt_head(self) -> Optional[nn.Module]:
        if not self.use_plddt_prediction_head:
            return None
        plddt_output_dim = 1 if self.plddt_prediction_mode == "regression" else self.plddt_num_bins
        return LinearClassificationHead(self.hidden_size, plddt_output_dim)

    def _build_aux_heads(self) -> Dict[str, nn.Module]:
        heads: Dict[str, nn.Module] = {}
        for track_name, n_bins in self.aux_track_num_bins.items():
            if n_bins < 2:
                raise ValueError(f"aux track '{track_name}' must have >=2 bins, got {n_bins}")
            heads[track_name] = LinearClassificationHead(self.hidden_size, n_bins)
        return heads
    
    def _freeze_base_model(self):
        """Freeze all parameters except LoRA adapters and classification head."""
        for name, param in self.model.named_parameters():
            # Keep LoRA parameters trainable
            if 'lora_' in name.lower():
                param.requires_grad = True
            # Keep classification head trainable
            elif 'classification_head' in name or 'classifier' in name or 'plddt_head' in name or 'aux_heads' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable:,} || Total: {total:,} || Trainable %: {100 * trainable / total:.2f}")
    
    def get_model(self) -> nn.Module:
        """Return the LoRA-wrapped model with classification head."""
        return self.model
    
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer

    def predict_from_fasta(
        self,
        input_fasta_path: str,
        output_fasta_path: str,
        model_checkpoint_path: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = None,
        max_tokens_per_batch: Optional[int] = None,
        device: Optional[str] = None,
        output_confidence_fasta: Optional[str] = None,
    ):
        """
        Predict 3Di sequences from amino acid FASTA using a T5-based model.

        Args:
            input_fasta_path: Path to input amino acid FASTA file
            output_fasta_path: Path to save predicted 3Di FASTA file
            model_checkpoint_path: Optional checkpoint with LoRA/heads
            batch_size: Batch size for inference
            max_length: Maximum sequence length for tokenization
            max_tokens_per_batch: Optional token budget for dynamic batching
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None.
            output_confidence_fasta: Optional path to write predicted pLDDT bins

        Returns:
            List of prediction records.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")

        if model_checkpoint_path:
            print(f"Loading model from: {model_checkpoint_path}")
            checkpoint = torch.load(model_checkpoint_path, map_location=device)
            print("\u2713 Checkpoint loaded")

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state_dict = checkpoint["model_state_dict"]
                if any(k.startswith("module.") for k in model_state_dict.keys()):
                    model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
                result = self.model.load_state_dict(model_state_dict, strict=False)
                epoch = checkpoint.get("epoch", "unknown")
                print(f"\u2713 Loaded model from epoch {epoch}")
                if result.missing_keys:
                    print(
                        f"  (Note: {len(result.missing_keys)} frozen weights not in checkpoint - this is expected)"
                    )
            else:
                if any(k.startswith("module.") for k in checkpoint.keys()):
                    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
                result = self.model.load_state_dict(checkpoint, strict=False)
                print("\u2713 Loaded model checkpoint")
                if result.missing_keys:
                    print(
                        f"  (Note: {len(result.missing_keys)} frozen weights not in checkpoint - this is expected)"
                    )

            label_vocab = checkpoint.get("label_vocab", [])
            if not label_vocab:
                label_vocab = list("ACDEFGHIKLMNPQRSTVWY")[: self.num_labels]
            idx2char = {i: c for i, c in enumerate(label_vocab)}
            args = checkpoint.get("args", {})
            plddt_label_vocab = checkpoint.get("plddt_label_vocab", args.get("plddt_label_vocab", []))
            if not plddt_label_vocab:
                plddt_label_vocab = [str(i) for i in range(10)]
            plddt_prediction_mode = args.get("plddt_prediction_mode", "classification")
        else:
            print("No checkpoint provided, using current model state")
            label_vocab = list("ACDEFGHIKLMNPQRSTVWY")[: self.num_labels]
            idx2char = {i: c for i, c in enumerate(label_vocab)}
            plddt_label_vocab = [str(i) for i in range(10)]
            plddt_prediction_mode = self.plddt_prediction_mode

        self.model = self.model.to(device)
        self.model.eval()

        tokenizer = self.tokenizer

        print(f"Reading input FASTA: {input_fasta_path}")
        aa_records = read_fasta(input_fasta_path)
        print(f"Found {len(aa_records)} sequences")

        def _iter_batches(records):
            batch = []
            token_budget = 0
            for header, seq in records:
                seq_len = len(seq)
                est_tokens = seq_len + 2  # account for special tokens
                if max_length is not None:
                    est_tokens = min(est_tokens, max_length)
                if batch:
                    over_size = len(batch) >= batch_size
                    over_tokens = max_tokens_per_batch is not None and (token_budget + est_tokens) > max_tokens_per_batch
                    if over_size or over_tokens:
                        yield batch
                        batch = []
                        token_budget = 0
                batch.append((header, seq))
                token_budget += est_tokens
            if batch:
                yield batch

        predictions = []
        processed = 0
        with torch.no_grad():
            for batch_records in _iter_batches(aa_records):
                headers, raw_seqs = zip(*batch_records)

                # T5 models require space-separated amino acids
                seqs = [" ".join(list(seq)) for seq in raw_seqs]

                enc = tokenizer(
                    list(seqs),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                )

                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                special_mask = enc["special_tokens_mask"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                plddt_logits = getattr(outputs, "plddt_logits", None)

                pred_labels = torch.argmax(logits, dim=-1).cpu()
                pred_plddt_labels = None
                pred_plddt_scores = None
                if plddt_logits is not None:
                    if plddt_prediction_mode == "regression":
                        pred_plddt_scores = torch.clamp(plddt_logits.squeeze(-1), min=0.0, max=100.0).cpu()
                    else:
                        pred_plddt_labels = torch.argmax(plddt_logits, dim=-1).cpu()

                for j, (header, raw_seq) in enumerate(zip(headers, raw_seqs)):
                    pred_3di = []
                    pred_plddt = []
                    pred_plddt_score = []
                    k = 0

                    for pos in range(pred_labels.shape[1]):
                        if special_mask[j, pos] == 0:
                            if k < len(raw_seq):
                                label_idx = pred_labels[j, pos].item()
                                pred_char = idx2char.get(label_idx, "X")
                                pred_3di.append(pred_char)

                                if pred_plddt_scores is not None:
                                    plddt_score = float(pred_plddt_scores[j, pos].item())
                                    pred_plddt_score.append(plddt_score)
                                    plddt_bin = max(0, min(9, int(plddt_score // 10.0)))
                                    pred_plddt.append(str(plddt_bin))
                                elif pred_plddt_labels is not None:
                                    plddt_idx = pred_plddt_labels[j, pos].item()
                                    pred_plddt.append(
                                        plddt_label_vocab[plddt_idx] if plddt_idx < len(plddt_label_vocab) else "0"
                                    )

                                k += 1

                    pred_3di_seq = "".join(pred_3di)
                    pred_plddt_seq = "".join(pred_plddt) if pred_plddt else None

                    if len(pred_3di_seq) != len(raw_seq):
                        print(
                            f"Warning: Length mismatch for {header}: AA={len(raw_seq)}, 3Di={len(pred_3di_seq)}"
                        )
                    if pred_plddt_seq is not None and len(pred_plddt_seq) != len(raw_seq):
                        print(
                            f"Warning: pLDDT length mismatch for {header}: AA={len(raw_seq)}, pLDDT={len(pred_plddt_seq)}"
                        )

                    predictions.append(
                        {
                            "header": header,
                            "aa_seq": raw_seq,
                            "pred_3di": pred_3di_seq,
                            "pred_plddt": pred_plddt_seq,
                            "pred_plddt_scores": pred_plddt_score if pred_plddt_score else None,
                        }
                    )

                processed += len(batch_records)
                if processed % (batch_size * 10) == 0 or processed >= len(aa_records):
                    print(f"Processed {processed}/{len(aa_records)} sequences")

                del outputs, logits, plddt_logits

        print(f"Writing predictions to: {output_fasta_path}")
        with open(output_fasta_path, "w") as f:
            for record in predictions:
                f.write(f">{record['header']}\n")
                pred_seq = record["pred_3di"]
                for i in range(0, len(pred_seq), 80):
                    f.write(pred_seq[i : i + 80] + "\n")

        if output_confidence_fasta:
            plddt_records = [record for record in predictions if record["pred_plddt"] is not None]
            if not plddt_records:
                print(
                    "Warning: pLDDT sidecar output requested, but this checkpoint/model has no pLDDT prediction head"
                )
            else:
                print(f"Writing predicted pLDDT bins to: {output_confidence_fasta}")
                with open(output_confidence_fasta, "w") as f:
                    for record in plddt_records:
                        f.write(f">{record['header']}\n")
                        pred_seq = record["pred_plddt"]
                        for i in range(0, len(pred_seq), 80):
                            f.write(pred_seq[i : i + 80] + "\n")

        print(f"\u2713 Prediction complete! {len(predictions)} sequences written to {output_fasta_path}")
        return predictions
    
    @staticmethod
    def tokenize_sequence(sequence: str, tokenizer, max_length: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Tokenize an amino acid sequence for T5 models.
        Uses space-separated amino acids.
        
        Args:
            sequence: Amino acid sequence (e.g., "MKTVF...")
            tokenizer: T5 tokenizer
            max_length: Maximum sequence length
        
        Returns:
            Dict with input_ids and attention_mask
        """
        # Space-separate amino acids
        spaced_seq = " ".join(list(sequence))
        
        # Tokenize
        encoded = tokenizer(
            spaced_seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        return encoded


# -----------------------------
# Utility Functions
# -----------------------------

def prepare_t5_batch(
    sequences: List[str],
    tokenizer,
    labels: Optional[List[str]] = None,
    label_vocab: Optional[Dict[str, int]] = None,
    max_length: int = 1024,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of sequences for T5 model training/inference.
    
    Args:
        sequences: List of amino acid sequences
        tokenizer: T5 tokenizer
        labels: Optional list of 3Di label sequences
        label_vocab: Mapping from 3Di characters to indices
        max_length: Maximum sequence length
        device: Target device
    
    Returns:
        Batch dict with input_ids, attention_mask, and optional labels
    """
    # Space-separate all sequences
    spaced_seqs = [" ".join(list(seq)) for seq in sequences]
    
    # Tokenize batch
    encoded = tokenizer(
        spaced_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device)
    }
    
    # Process labels if provided
    if labels is not None and label_vocab is not None:
        label_ids = []
        for label_seq, input_ids in zip(labels, encoded["input_ids"]):
            seq_labels = []
            char_idx = 0
            for token_id in input_ids:
                token = tokenizer.decode([token_id])
                # Check if this is an amino acid token
                if token.strip() in "ACDEFGHIKLMNPQRSTVWY":
                    if char_idx < len(label_seq):
                        label_char = label_seq[char_idx].upper()
                        seq_labels.append(label_vocab.get(label_char, -100))
                        char_idx += 1
                    else:
                        seq_labels.append(-100)
                else:
                    seq_labels.append(-100)  # Ignore special tokens
            label_ids.append(seq_labels)
        
        batch["labels"] = torch.tensor(label_ids, dtype=torch.long, device=device)
    
    return batch

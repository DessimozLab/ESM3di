"""Supported ESM++ LoRA/CNN runtime; retired architectures are archived."""
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Tuple, Optional
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.pytorch_utils import Conv1D
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

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


def _extract_sequence_output(outputs):
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    return outputs.last_hidden_state


class ESMWithCNNHead(nn.Module):
    """
    Wrapper that replaces the standard classifier with a CNN head.
    Note: Loss is computed externally in training loop to support Focal Loss etc.
    """
    def __init__(self, base_model, cnn_head):
        super().__init__()
        self.base_model = base_model
        self.cnn_head = cnn_head

        # Store config for compatibility
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        labels = kwargs.pop("labels", None)

        # Get encoder outputs (without classification head)
        # Use output_hidden_states to get the last hidden state
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

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)


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
    Classifiers and heads are filtered before applying the adapters.
    """
    # Modules to exclude from LoRA (these train fully or don't work well with LoRA)
    exclude_patterns = [ 'gate']
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
    """
    for name, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if (
            "lora_" in name
            or "classifier" in name
            or "cnn_head" in name
        ):
            p.requires_grad = True


class ESM3DiModel:
    """ESM++ backbone with LoRA and the supported per-residue CNN head."""

    def __init__(
        self, hf_model_name: str, num_labels: int, lora_r: int = 8,
        lora_alpha: float = 16.0, lora_dropout: float = 0.05,
        target_modules=None, use_cnn_head: bool = True, cnn_num_layers: int = 2,
        cnn_kernel_size: int = 3, cnn_dropout: float = 0.1,
        revision: Optional[str] = None, backbone_init: Optional[str] = None,
    ):
        from .checkpoints import resolve_hf_revision, get_tokenizer

        if not use_cnn_head:
            raise ValueError("Only the ESM++ CNN architecture is supported")
        if not os.path.isdir(hf_model_name) and not hf_model_name.startswith("Synthyra/ESMplusplus_"):
            raise ValueError("Only ESM++ backbones are supported; retired architectures are in experiments/legacy_runtime")
        self.hf_model_name = hf_model_name
        self.num_labels = num_labels
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_cnn_head = True
        self.cnn_num_layers = cnn_num_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_dropout = cnn_dropout
        self.is_t5 = False
        self.revision = resolve_hf_revision(hf_model_name, revision)
        self.backbone_init = backbone_init
        self.backbone_transfer_report = None
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            hf_model_name, num_labels=num_labels, trust_remote_code=True,
            revision=self.revision,
        )
        self.tokenizer = get_tokenizer(self.base_model, hf_model_name, self.revision)
        if backbone_init:
            from .checkpoints import load_mlm_state_dict, transfer_backbone_weights

            self.backbone_transfer_report = transfer_backbone_weights(
                self.base_model, load_mlm_state_dict(backbone_init, revision=self.revision),
            )
            print(f"MLM backbone transfer: {self.backbone_transfer_report.summary()}")
        self.target_modules = target_modules or discover_lora_target_modules(self.base_model)
        self._setup_lora()
        self._setup_cnn_head()
        self.freeze_base_model()


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

        task_type = TaskType.TOKEN_CLS

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
        )


    def freeze_base_model(self):
        """Freeze all base model parameters except LoRA and classifier."""
        freeze_all_but_lora_and_classifier(self.model)


    def get_model(self):
        """Return the LoRA-wrapped model."""
        return self.model


    def predict_from_fasta(
        self, input_fasta_path: str, output_fasta_path: str,
        model_checkpoint_path: Optional[str] = None, batch_size: int = 4,
        device: Optional[str] = None, output_confidence_fasta: Optional[str] = None,
    ):
        """Legacy convenience API; use the inference CLI for windowed prediction."""
        from .checkpoints import load_state_into_model, lora_scaling_from_checkpoint, strip_module_prefix

        if output_confidence_fasta:
            raise ValueError("The supported CNN model does not predict pLDDT")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        label_vocab = list("ACDEFGHIKLMNPQRSTVWY")[:self.num_labels]
        if model_checkpoint_path:
            checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint)
            load_state_into_model(
                self.model, strip_module_prefix(state),
                lora_scaling=lora_scaling_from_checkpoint(checkpoint),
            )
            label_vocab = checkpoint.get("label_vocab", label_vocab)
        self.model.to(device).eval()
        records = read_fasta(input_fasta_path)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                encoded = self.tokenizer(
                    [sequence for _, sequence in batch], return_tensors="pt",
                    padding=True, add_special_tokens=True, return_special_tokens_mask=True,
                )
                labels = self.model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                ).logits.argmax(dim=-1).cpu()
                mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
                for i, (header, sequence) in enumerate(batch):
                    prediction = "".join(label_vocab[index] for index in labels[i][mask[i]].tolist())
                    if len(prediction) != len(sequence):
                        raise RuntimeError(f"Prediction length mismatch for {header}")
                    predictions.append(dict(
                        header=header, aa_seq=sequence, pred_3di=prediction,
                        pred_plddt=None, pred_plddt_scores=None,
                    ))
        with open(output_fasta_path, "w") as handle:
            for record in predictions:
                handle.write(f">{record['header']}\n{record['pred_3di']}\n")
        return predictions

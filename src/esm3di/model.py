import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModel
from transformers.pytorch_utils import Conv1D
from peft import get_peft_model, LoraConfig, TaskType

from .model_outputs import TokenClassifierOutputWithPLDDT, PLDDT_BIN_VOCAB
from .iterative_head import IterativeTransformerClassificationHead, ESMWithIterativeTransformerHead



class CNNClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 2, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        x = self.cnn_layers(hidden_states.transpose(1, 2)).transpose(1, 2)
        return self.classifier(x)


class TransformerClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, transformer_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.input_projection = nn.Identity() if hidden_size == transformer_dim else nn.Linear(hidden_size,
                                                                                               transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads,
                                                   dim_feedforward=transformer_dim * 4, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        self.classifier = nn.Linear(transformer_dim, num_labels)

    def forward(self, hidden_states, attention_mask=None):
        x = self.input_projection(hidden_states)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        return self.classifier(self.norm(self.encoder(x, src_key_padding_mask=src_key_padding_mask) + x))


class LinearClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask=None):
        return self.classifier(hidden_states)


def _extract_sequence_output(outputs):
    return outputs.hidden_states[-1] if hasattr(outputs,
                                                'hidden_states') and outputs.hidden_states is not None else outputs.last_hidden_state


class ESMWithCustomHeadBase(nn.Module):
    """Base class for attaching custom heads to ESM."""

    def __init__(self, base_model, head_module, plddt_head=None, aux_heads=None):
        super().__init__()
        self.base_model = base_model
        self.custom_head = head_module
        self.plddt_head = plddt_head
        self.aux_heads = nn.ModuleDict(aux_heads or {})
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        labels = kwargs.pop("labels", None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                  **kwargs)
        sequence_output = _extract_sequence_output(outputs)

        # Determine if head needs attention mask (Transformer) or not (CNN/Linear)
        if isinstance(self.custom_head, TransformerClassificationHead):
            logits = self.custom_head(sequence_output, attention_mask=attention_mask)
        else:
            logits = self.custom_head(sequence_output)

        plddt_logits = self.plddt_head(sequence_output) if self.plddt_head else None
        aux_logits = {name: head(sequence_output) for name, head in self.aux_heads.items()} if self.aux_heads else None

        loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, self.config.num_labels),
                                                      labels.view(-1)) if labels is not None else None

        return TokenClassifierOutputWithPLDDT(loss=loss, logits=logits,
                                              hidden_states=getattr(outputs, 'hidden_states', None),
                                              plddt_logits=plddt_logits, aux_logits=aux_logits)


class ESM3DiModel:
    """Core Model class establishing the architecture and LoRA wrappers."""

    def __init__(self, hf_model_name: str, num_labels: int, **kwargs):
        self.hf_model_name = hf_model_name
        self.num_labels = num_labels
        self.kwargs = kwargs
        self._load_model()
        self._setup_lora()
        self._attach_heads()
        self.freeze_base_model()

    def _load_model(self):
        try:
            self.base_model = AutoModelForTokenClassification.from_pretrained(self.hf_model_name,
                                                                              num_labels=self.num_labels,
                                                                              trust_remote_code=True)
        except Exception:
            self.base_model = AutoModel.from_pretrained(self.hf_model_name, trust_remote_code=True)
            self.tokenizer = self.base_model.tokenizer

    def _setup_lora(self):
        target_modules = self.kwargs.get('target_modules') or [m for m in self._discover_lora() if not any(
            x in m.lower() for x in ['classifier', 'head', 'pooler', 'lm_head'])]
        lora_config = LoraConfig(task_type=TaskType.TOKEN_CLS, r=self.kwargs.get('lora_r', 8),
                                 lora_alpha=self.kwargs.get('lora_alpha', 16), target_modules=target_modules)
        self.model = get_peft_model(self.base_model, lora_config)

    def _attach_heads(self):
        hidden_size = self.base_model.config.hidden_size
        plddt_head = LinearClassificationHead(hidden_size, self.kwargs.get('plddt_num_bins', 10)) if self.kwargs.get(
            'use_plddt_prediction_head') else None

        if self.kwargs.get('use_cnn_head'):
            head = CNNClassificationHead(hidden_size, self.num_labels, self.kwargs.get('cnn_num_layers', 2),
                                         self.kwargs.get('cnn_kernel_size', 3))
        elif self.kwargs.get('use_transformer_head'):
            head = TransformerClassificationHead(hidden_size, self.num_labels,
                                                 self.kwargs.get('transformer_head_dim', 256))
        else:
            head = LinearClassificationHead(hidden_size, self.num_labels)

        self.model = ESMWithCustomHeadBase(self.model, head, plddt_head=plddt_head)

    def _discover_lora(self):
        targets = set()
        for name, child in self.base_model.named_modules():
            if isinstance(child, (nn.Linear, nn.Embedding, nn.Conv2d, Conv1D)) and 'gate' not in name.lower():
                targets.add(name)
        return list(targets)

    def freeze_base_model(self):
        for name, p in self.model.named_parameters():
            p.requires_grad = any(x in name for x in
                                  ["lora_", "classifier", "plddt_head", "aux_heads", "cnn_head", "transformer_head",
                                   "custom_head"])

    def get_model(self):
        return self.model
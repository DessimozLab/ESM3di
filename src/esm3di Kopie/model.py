import torch
import torch.nn as nn
from transformers import AutoModel
from .model_outputs import TokenClassifierOutputWithPLDDT
from .iterative_head import IterativeTransformerClassificationHead


class CNNClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 2, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        x = self.cnn_layers(hidden_states.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(x)


class TransformerClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, transformer_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.input_projection = nn.Identity() if hidden_size == transformer_dim else nn.Linear(hidden_size, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dim_feedforward=transformer_dim * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        self.out_proj = nn.Linear(transformer_dim, num_labels)

    def forward(self, hidden_states, attention_mask=None):
        x = self.input_projection(hidden_states)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        return self.out_proj(self.norm(self.encoder(x, src_key_padding_mask=src_key_padding_mask) + x))


class ESM3DiModel(nn.Module):
    """Core Custom Architecture containing the ESM backbone and structural heads."""

    def __init__(self, hf_model_name: str, num_labels: int, **kwargs):
        super().__init__()

        self.esm = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
        if hasattr(self.esm, "pooler"):
            self.esm.pooler = None
        hidden_size = self.esm.config.hidden_size

        # 2. Main 3Di Classification Head Router
        if kwargs.get('use_cnn_head'):
            self.classifier = CNNClassificationHead(
                hidden_size,
                num_labels,
                kwargs.get('cnn_num_layers', 2),
                kwargs.get('cnn_kernel_size', 3),
                kwargs.get('cnn_dropout', 0.1),
            )

        elif kwargs.get('use_transformer_head'):
            self.classifier = TransformerClassificationHead(
                hidden_size,
                num_labels,
                kwargs.get('transformer_head_dim', 256),
                kwargs.get('transformer_head_layers', 2),
                kwargs.get('transformer_head_dropout', 0.1),
                kwargs.get('transformer_head_num_heads', None),
            )

        elif kwargs.get('use_iterative_transformer_head'):
            self.classifier = IterativeTransformerClassificationHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                transformer_dim=kwargs.get('transformer_head_dim', 256),
                num_layers=kwargs.get('transformer_head_layers', 2),
                dropout=kwargs.get('transformer_head_dropout', 0.1),
                num_heads=kwargs.get('transformer_head_num_heads', None),
                use_positional_encoding=kwargs.get('use_positional_encoding', True),
                use_hidden_state_feedback=kwargs.get('use_hidden_state_feedback', True),
                use_gru_gate=kwargs.get('use_gru_gate', False),
                max_iterations=kwargs.get('iterative_head_max_iterations', 5),
                halt_threshold=kwargs.get('iterative_head_halt_threshold', 0.95)
            )
        else:
            self.classifier = nn.Linear(hidden_size, num_labels)

        self.plddt_head = nn.Linear(hidden_size, kwargs.get('plddt_num_bins', 10)) if kwargs.get('use_plddt_prediction_head') else None

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state

        # Both transformer-based heads require the attention mask for padding safety
        if isinstance(self.classifier, (TransformerClassificationHead, IterativeTransformerClassificationHead)):
            logits = self.classifier(sequence_output, attention_mask=attention_mask)
        else:
            logits = self.classifier(sequence_output)

        plddt_logits = self.plddt_head(sequence_output) if self.plddt_head else None

        return TokenClassifierOutputWithPLDDT(logits=logits, plddt_logits=plddt_logits)
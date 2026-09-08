"""Module defining the structural architecture of the ESM sequence modeling pipeline.

Includes the custom 1D CNN sequence-labeling head and the integrated top-level
architectural wrapper container matching the training layout.
"""

from typing import Any, Optional
import torch
import torch.nn as nn

# Notice: 'AutoModel' and 'Optional' imports kept exactly as requested
from transformers import AutoModel


class CNNClassificationHead(nn.Module):
    """Custom 1D Convolutional classification head with fixed hyperparameters.

    Processes token representations from a transformer backbone along the sequence
    length axis using multi-layer 1D convolutions and projects them to 3Di vocabulary space.
    """

    def __init__(self, hidden_size: int):
        """Initializes the CNNClassificationHead with standard production parameters.

        Args:
            hidden_size: Dimensionality of the incoming sequence token embeddings.
        """
        super().__init__()
        num_labels = 20
        num_layers = 2
        kernel_size = 3
        dropout = 0.1

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.cnn_layers = nn.Sequential(*layers)

        # Named 'classifier' to match original checkpoint state_dict keys perfectly
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies 1D convolutions across token embeddings and projects to logits.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Logits tensor of shape (batch_size, seq_len, num_labels).
        """
        # Shape: (B, L, H) -> (B, H, L) for Conv1D -> (B, L, H) for Linear Classifier
        x = self.cnn_layers(hidden_states.transpose(1, 2)).transpose(1, 2)
        return self.classifier(x)


class InferenceOutput:
    """Wrapper class mimicking standard Hugging Face model output objects."""

    def __init__(self, logits: torch.Tensor):
        """Initializes the output container.

        Args:
            logits: Output classification logits tensor.
        """
        self.logits = logits


class ESMWithCNNHead(nn.Module):
    """Integrated top-level wrapper class for the ESM-CNN architecture.

    Encapsulates the base PEFT-wrapped transformer backbone and routes sequence
    hidden states cleanly into the specialized Convolutional head.
    """

    def __init__(self, peft_model: nn.Module, cnn_head: nn.Module):
        """Initializes the unified network structure.

        Args:
            peft_model: The PEFT-wrapped AutoModelForTokenClassification instance.
            cnn_head: The custom CNNClassificationHead instance.
        """
        super().__init__()
        self.base_model = peft_model
        self.cnn_head = cnn_head

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs: Any
    ) -> InferenceOutput:
        """Runs the model forward pass to generate structural sequence predictions.

        Args:
            input_ids: Encoded token IDs of shape (batch_size, seq_len).
            attention_mask: Mask to bypass padding tokens.
            **kwargs: Extra parameters passed directly to the base transformer backbone.

        Returns:
            An InferenceOutput instance containing the sequence logits.
        """
        # Pass through the base PEFT-wrapped model (which is ESMplusplusModel + LoRA layers)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # Pull out the hidden representation from the final layers
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            sequence_output = outputs.hidden_states[-1]
        else:
            sequence_output = outputs.last_hidden_state

        logits = self.cnn_head(sequence_output)

        return InferenceOutput(logits=logits)
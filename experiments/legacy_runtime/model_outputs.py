from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers.utils import ModelOutput


PLDDT_BIN_VOCAB = [str(i) for i in range(10)]


@dataclass
class TokenClassifierOutputWithPLDDT(ModelOutput):
    """Token-classification output with optional auxiliary token tracks."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    plddt_logits: Optional[torch.Tensor] = None
    aux_logits: Optional[Dict[str, torch.Tensor]] = None

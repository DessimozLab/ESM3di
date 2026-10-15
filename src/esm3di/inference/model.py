"""Release architecture backed by the same CNN implementation as training."""

from .._core.model import CNNClassificationHead as _CNNClassificationHead
from .._core.model import ESMWithCNNHead as _ESMWithCNNHead


class CNNClassificationHead(_CNNClassificationHead):
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size, num_labels=20, num_layers=2, kernel_size=3, dropout=0.1)


class ESMWithCNNHead(_ESMWithCNNHead):
    def __init__(self, peft_model, cnn_head):
        super().__init__(base_model=peft_model, cnn_head=cnn_head)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        kwargs.setdefault("return_dict", True)
        return super().forward(input_ids, attention_mask, **kwargs)

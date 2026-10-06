import torch
from torch import nn

from esm3di.model import CNNClassificationHead, ESMWithCNNHead


def test_cnn_classification_head_returns_20_logits_per_token():
    head = CNNClassificationHead(hidden_size=4)
    hidden_states = torch.randn(2, 7, 4)

    logits = head(hidden_states)

    assert logits.shape == (2, 7, 20)


class _BackboneWithHiddenStates(nn.Module):
    def forward(self, **kwargs):
        return type("Output", (), {"hidden_states": (torch.ones(1, 3, 4),)})()


class _BackboneWithLastHiddenState(nn.Module):
    def forward(self, **kwargs):
        return type("Output", (), {"last_hidden_state": torch.ones(1, 3, 4)})()


class _RecordingHead(nn.Module):
    def forward(self, hidden_states):
        assert hidden_states.shape == (1, 3, 4)
        return hidden_states.sum(dim=-1, keepdim=True)


def test_esm_wrapper_prefers_final_hidden_state():
    wrapper = ESMWithCNNHead(_BackboneWithHiddenStates(), _RecordingHead())

    output = wrapper(torch.ones(1, 3, dtype=torch.long))

    assert output.logits.shape == (1, 3, 1)
    assert torch.equal(output.logits, torch.full((1, 3, 1), 4.0))


def test_esm_wrapper_falls_back_to_last_hidden_state():
    wrapper = ESMWithCNNHead(_BackboneWithLastHiddenState(), _RecordingHead())

    output = wrapper(torch.ones(1, 3, dtype=torch.long))

    assert torch.equal(output.logits, torch.full((1, 3, 1), 4.0))

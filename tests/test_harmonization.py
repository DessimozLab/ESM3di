import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from esm3di import checkpoints
from esm3di.inference.model import CNNClassificationHead
from esm3di.inference.predictor import ESM3DiPredictor
from esm3di.training.cli import build_parser
from esm3di.ESM3di_model import CNNClassificationHead as TrainingCNN


@pytest.mark.parametrize("name", ["train_mlm", "train_3di", "prepare_data"])
def test_legacy_module_is_same_implementation(name):
    assert importlib.import_module(f"esm3di.{name}") is importlib.import_module(
        f"esm3di.training.{name}"
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [("predict", "checkpoint_predictor"), ("cutting_cli", "cutting"), ("foldseek_db", "foldseek_db")],
)
def test_legacy_inference_modules_are_same_implementation(old, new):
    assert importlib.import_module(f"esm3di.{old}") is importlib.import_module(
        f"esm3di.inference.{new}"
    )


def test_learning_cli_has_no_inference_commands():
    parser = build_parser()
    for command in ("prepare", "mlm", "train", "pipeline"):
        assert command in parser.format_help()
    assert "predict" not in parser.format_help()


def test_inference_release_defaults_remain_public():
    from esm3di.inference import DEFAULT_HF_REPO, DEFAULT_REVISION, VOCAB_3DI

    assert DEFAULT_HF_REPO == "cactuskid13/ESM3di_Small_MLM_3di"
    assert DEFAULT_REVISION == "46c5f7d"
    assert VOCAB_3DI == list("ACDEFGHIKLMNPQRSTVWY")


def test_legacy_cli_keeps_prediction():
    from esm3di.cli import build_parser as legacy_parser

    args = legacy_parser().parse_args(
        ["predict", "--checkpoint", "best.pt", "-i", "in.fasta", "-o", "out.fasta"]
    )
    assert args.command == "predict"


def test_inference_import_does_not_import_trainers():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from esm3di.inference import ESM3DiPredictor; "
         "assert 'esm3di.training.train_3di' not in sys.modules; "
         "assert 'esm3di.training.train_mlm' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_shared_cnn_preserves_state_dict_and_exact_logits():
    torch.manual_seed(9)
    release = CNNClassificationHead(8).eval()
    training = TrainingCNN(8, 20, num_layers=2, kernel_size=3, dropout=0.1).eval()
    training.load_state_dict(release.state_dict(), strict=True)
    hidden = torch.randn(2, 11, 8)
    assert list(release.state_dict()) == list(training.state_dict())
    assert torch.equal(release(hidden), training(hidden))


def test_training_checkpoint_uses_shared_loader_and_saved_vocabulary(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    path.touch()
    model = nn.Linear(2, 2)
    tokenizer = object()
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(model=model, tokenizer=tokenizer), {"label_vocab": ["T", "A"]}

    monkeypatch.setattr(checkpoints, "load_3di_checkpoint", load)
    predictor = ESM3DiPredictor(path, device="cpu")
    assert predictor.model is model
    assert predictor.tokenizer is tokenizer
    assert predictor.idx2char == {0: "T", 1: "A"}
    assert calls == [(str(path), {"device": "cpu", "hf_revision": None})]


def test_missing_hf_cnn_is_an_error(tmp_path, monkeypatch):
    from esm3di.inference import predictor as module

    config = SimpleNamespace(hidden_size=8)
    backbone = nn.Linear(8, 8)
    backbone.config = config
    backbone.tokenizer = object()
    monkeypatch.setattr(module.PeftConfig, "from_pretrained",
                        lambda path: SimpleNamespace(base_model_name_or_path="fake"))
    monkeypatch.setattr(module.AutoModel, "from_pretrained", lambda *args, **kwargs: backbone)
    monkeypatch.setattr(module.PeftModel, "from_pretrained", lambda base, path: base)
    with pytest.raises(FileNotFoundError, match="cnn_head.bin"):
        ESM3DiPredictor(tmp_path, device="cpu")


def test_mlm_transfer_copies_backbone_but_not_classifier():
    target = nn.Module()
    target.embed = nn.Embedding(3, 2)
    target.classifier = nn.Linear(2, 2)
    classifier = target.classifier.weight.detach().clone()
    report = checkpoints.transfer_backbone_weights(
        target, {"embed.weight": torch.ones(3, 2), "classifier.weight": torch.zeros(2, 2)}
    )
    assert report.copied == 1
    assert torch.equal(target.embed.weight, torch.ones(3, 2))
    assert torch.equal(target.classifier.weight, classifier)


def test_mlm_transfer_rejects_missing_backbone():
    with pytest.raises(RuntimeError, match="Only 0/"):
        checkpoints.transfer_backbone_weights(nn.Linear(2, 2), {})


def test_windowed_checkpoint_predictor_keeps_every_residue(monkeypatch):
    from esm3di.inference.checkpoint_predictor import Predictor

    predictor = Predictor.__new__(Predictor)
    predictor.window = 8
    predictor.overlap = 2
    predictor.label_vocab = ["A", "T"]

    def probs(chunks):
        return [torch.tensor([[1.0, 0.0]] * len(chunk)) for chunk in chunks]

    monkeypatch.setattr(predictor, "_probs", probs)
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    result = list(predictor.predict([("protein", sequence)], progress=False))
    assert result[0][0] == 0
    assert result[0][1] == "A" * len(sequence)
    assert len(result[0][2]) == len(sequence)

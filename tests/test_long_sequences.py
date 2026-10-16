from types import SimpleNamespace

import numpy as np
import pytest
import torch

from esm3di.inference import cli, cutting
from esm3di.inference.io import read_fasta
from esm3di.inference.predictor import ESM3DiPredictor
from esm3di.inference.windowing import residue_window_size, windowed_outputs


class ResidueTokenizer:
    model_max_length = 10

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, sequences, **kwargs):
        assert kwargs["truncation"] is False
        assert max(map(len, sequences)) <= self.model_max_length - 2
        length = max(map(len, sequences)) + 2
        ids, attention, special = [], [], []
        for seq in sequences:
            padding = length - len(seq) - 2
            ids.append([0] + [1 if char == "A" else 2 for char in seq] + [0] * (padding + 1))
            attention.append([1] * (len(seq) + 2) + [0] * padding)
            special.append([1] + [0] * len(seq) + [1] * (padding + 1))
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attention),
            "special_tokens_mask": torch.tensor(special),
        }


class ResidueModel:
    def __call__(self, input_ids, attention_mask):
        logits = torch.stack((input_ids == 1, input_ids == 2), dim=-1).float() * 4
        return SimpleNamespace(logits=logits, hidden_states=(input_ids.unsqueeze(-1).float(),))


@pytest.fixture
def predictor():
    result = ESM3DiPredictor.__new__(ESM3DiPredictor)
    result.device = torch.device("cpu")
    result.tokenizer = ResidueTokenizer()
    result.model = ResidueModel()
    result.truncate = False
    result.idx2char = {0: "A", 1: "T"}
    return result


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_full_length_predictions_include_tail_and_preserve_order(predictor, batch_size):
    sequences = ["A" * 21 + "T" * 12, "AT", "", "AT" * 12, "T" * 8, "A" * 9]
    assert predictor.predict_batch(sequences, batch_size=batch_size) == sequences
    records = [(str(idx), seq) for idx, seq in enumerate(sequences)]
    assert predictor.predict_records(records, batch_size=batch_size) == records


def test_short_predictions_match_unwindowed_logits(predictor):
    sequences = ["ATATA", "TT"]
    enc = predictor.tokenizer(sequences, truncation=False)
    logits = predictor.model(enc["input_ids"], enc["attention_mask"]).logits
    expected = [
        "".join(predictor.idx2char[i] for i in logits[row, 1:len(seq) + 1].argmax(-1).tolist())
        for row, seq in enumerate(sequences)
    ]
    assert predictor.predict_batch(sequences) == expected


def test_overlap_is_averaged_not_overwritten():
    tokenizer = ResidueTokenizer()

    def positions(ids, attention):
        return torch.arange(ids.shape[1]).expand_as(ids).unsqueeze(-1).float()

    [actual] = list(windowed_outputs(["A" * 12], tokenizer, "cpu", 2, positions))
    expected = torch.tensor([1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8]).float()
    torch.testing.assert_close(actual[:, 0], expected)


def test_truncation_requires_opt_in(predictor):
    sequence = "A" * 8 + "T" * 9
    assert predictor.predict(sequence) == sequence
    predictor.truncate = True
    assert predictor.predict(sequence) == sequence[:8]


def test_tokenizer_mismatch_fails_explicitly(predictor):
    class BrokenTokenizer(ResidueTokenizer):
        def __call__(self, *args, **kwargs):
            enc = super().__call__(*args, **kwargs)
            enc["special_tokens_mask"][:, 1] = 1
            return enc

    predictor.tokenizer = BrokenTokenizer()
    with pytest.raises(RuntimeError, match="refusing incomplete predictions"):
        predictor.predict("AT" * 20)


def test_context_limit_and_invalid_batch_size(predictor):
    predictor.tokenizer.model_max_length = 10**30
    assert residue_window_size(predictor.tokenizer) == 1022
    with pytest.raises(ValueError, match="batch_size"):
        predictor.predict_batch(["A"], batch_size=0)


def test_cutter_embeds_every_residue(predictor):
    sequences = ["A" * 25 + "T" * 10, "TT"]
    matrices = cutting.embed_sequences(
        predictor.model, predictor.tokenizer, sequences, False, "cpu", batch_size=2,
    )
    for seq, matrix in zip(sequences, matrices):
        assert matrix.shape == (len(seq), 1)
        np.testing.assert_equal(matrix[:, 0], [1 if char == "A" else 2 for char in seq])


@pytest.mark.parametrize("truncate", [False, True])
def test_cli_prediction_and_confidence_cover_expected_residues(
    predictor, tmp_path, monkeypatch, truncate,
):
    def load(*args, **kwargs):
        predictor.truncate = kwargs["truncate"]
        return predictor

    monkeypatch.setattr(cli.ESM3DiPredictor, "from_pretrained", load)
    source = tmp_path / "input.fa"
    sequence = "A" * 15 + "T" * 10
    source.write_text(f">protein description\n{sequence}\n")
    output = tmp_path / "prediction.fa"
    confidence = tmp_path / "confidence.tsv"
    extra = ["--truncate"] if truncate else []
    cli.main(["predict", "--input-fasta", str(source), "--output-fasta", str(output),
              "--num-gpus", "1", *extra])
    expected = sequence[:8] if truncate else sequence
    assert read_fasta(output) == [("protein description", expected)]
    cli.main(["perplexity", "--input-fasta", str(source), "--output-tsv", str(confidence), *extra])
    rows = [row.split("\t") for row in confidence.read_text().splitlines()[1:]]
    assert [int(row[1]) for row in rows] == list(range(1, len(expected) + 1))
    assert "".join(row[2] for row in rows) == expected
    assert all(1 <= float(row[3]) <= 2 for row in rows)


def test_foldseek_database_has_full_length_paired_records(predictor, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.ESM3DiPredictor, "from_pretrained", lambda *args, **kwargs: predictor)
    source = tmp_path / "input.fa"
    sequence = "A" * 15 + "T" * 10
    source.write_text(f">protein\n{sequence}\n")
    output = tmp_path / "db"
    cli.main(["foldseek-db", "--input-fasta", str(source), "--output-db", str(output),
              "--num-gpus", "1"])
    assert output.read_bytes() == sequence.encode() + b"\n\0"
    assert (tmp_path / "db_ss").read_bytes() == sequence.encode() + b"\n\0"


def test_cut_pipeline_reconstructs_long_input(predictor, tmp_path, monkeypatch):
    sequence = "A" * 15 + "T" * 10
    source = tmp_path / "input.fa"
    source.write_text(f">protein\n{sequence}\n")
    monkeypatch.setattr(cutting, "load_esm3di_embedding_model",
                        lambda **kwargs: (predictor.model, predictor.tokenizer, False))
    monkeypatch.setattr(cutting, "load_ffn_classifier", lambda *args: (object(), {}))
    monkeypatch.setattr(cutting, "predict_ffn_probs",
                        lambda model, emb, device: np.where(emb[:, 0] == 1, 0.8, 0.2))
    outputs = cutting.run_cutting_pipeline(
        str(source), "model.pt", "ffn.pt", None, str(tmp_path / "cut"),
        classifier_mode="ffn", device="cpu",
    )
    assert "".join(seq for _, seq in read_fasta(outputs["products_fasta"])) == sequence
    assert len(outputs["residue_csv"].read_text().splitlines()) == len(sequence) + 1


@pytest.mark.parametrize("truncate", [False, True])
def test_multi_gpu_worker_preserves_truncation_setting_and_revision(monkeypatch, truncate):
    import queue
    import threading
    from esm3di.inference import predictor as module

    calls = []

    class WorkerPredictor:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def predict_fasta(self, *args, **kwargs):
            assert kwargs["num_gpus"] == 1

    monkeypatch.setattr(module, "ESM3DiPredictor", WorkerPredictor)
    progress, error = queue.Queue(), threading.Event()
    module._gpu_worker(2, "input.fa", "output.fa", "owner/repo", 1, progress, error,
                       truncate=truncate, revision="pinned")
    assert not error.is_set()
    assert progress.get_nowait() == ("done", 2)
    assert calls == [{"device": "cuda:2", "truncate": truncate, "revision": "pinned"}]

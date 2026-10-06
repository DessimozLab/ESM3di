import pytest

from esm3di.inference import ESM3DiPredictor
from esm3di.io import read_fasta


def test_predict_records_preserves_order_when_sorting(monkeypatch):
    predictor = ESM3DiPredictor.__new__(ESM3DiPredictor)
    calls = []

    def fake_predict_batch(sequences, batch_size):
        calls.append((sequences, batch_size))
        return [sequence.lower() for sequence in sequences]

    monkeypatch.setattr(predictor, "predict_batch", fake_predict_batch)
    records = [("long", "ABCDE"), ("short", "AB")]

    result = predictor.predict_records(records, batch_size=2, sort_by_length=True)

    assert result == [("long", "abcde"), ("short", "ab")]
    assert calls == [(["AB", "ABCDE"], 2)]


def test_predict_records_rejects_missing_predictions(monkeypatch):
    predictor = ESM3DiPredictor.__new__(ESM3DiPredictor)
    monkeypatch.setattr(predictor, "predict_batch", lambda sequences, batch_size: [])

    with pytest.raises(RuntimeError, match="Prediction count"):
        predictor.predict_records([("id", "AC")])


def test_predict_fasta_reads_predicts_and_writes_records(monkeypatch, tmp_path):
    predictor = ESM3DiPredictor.__new__(ESM3DiPredictor)
    input_path = tmp_path / "input.fasta"
    output_path = tmp_path / "nested" / "output.fasta"
    input_path.write_text(">first description\nACD\n>second\nEF\n")

    monkeypatch.setattr(
        predictor,
        "predict_records",
        lambda records, batch_size, sort_by_length: [
            (header, "XYZ"[:len(sequence)])
            for header, sequence in records
        ],
    )

    predictor.predict_fasta(input_path, output_path, batch_size=4, sort_by_length=False)

    assert read_fasta(output_path) == [
        ("first description", "XYZ"),
        ("second", "XY"),
    ]

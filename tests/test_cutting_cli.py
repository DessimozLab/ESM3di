import pandas as pd
import pytest
import torch
import numpy as np

from esm3di.inference import cli, cutting


def arguments(tmp_path, classifier="ensemble"):
    return [
        "--input-fasta", str(tmp_path / "input.fa"),
        "--esm3di-checkpoint", "embedding.pt",
        "--ffn-checkpoint", "ffn.pt",
        "--transformer-checkpoint", "transformer.pt",
        "--classifier", classifier, "--cutoff", "0.6",
        "--output-prefix", str(tmp_path / "products"),
        "--device", "cpu", "--batch-size", "2", "--revision", "revision",
    ]


@pytest.mark.parametrize("classifier", ["ffn", "transformer", "ensemble"])
def test_inference_cut_matches_standalone_arguments(tmp_path, monkeypatch, classifier):
    calls = []
    monkeypatch.setattr(cutting, "run_cutting_pipeline", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        cli.ESM3DiPredictor, "from_pretrained",
        lambda *args, **kwargs: pytest.fail("cut must not load the 3Di predictor"),
    )
    args = arguments(tmp_path, classifier)
    cutting.main(args)
    cli.main(["cut", *args])
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["classifier_mode"] == classifier
    assert calls[0]["revision"] == "revision"


def test_cut_errors_are_reported(tmp_path, monkeypatch, capsys):
    def fail(**kwargs):
        raise FileNotFoundError("Missing embedding checkpoint")

    monkeypatch.setattr(cutting, "run_cutting_pipeline", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["cut", *arguments(tmp_path)])
    assert error.value.code == 1
    assert "Missing embedding checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("output_format", ["fasta", "csv", "both"])
def test_cut_produces_probabilities_summary_and_products(tmp_path, monkeypatch, output_format):
    (tmp_path / "input.fa").write_text(">protein\nACDE\n")
    monkeypatch.setattr(cutting, "load_esm3di_embedding_model", lambda **kwargs: (None, None, False))
    monkeypatch.setattr(
        cutting, "embed_sequences", lambda **kwargs: [np.zeros((4, 2), dtype=np.float32)],
    )
    head = cutting.SmallCutSiteFFN(2)
    for parameter in head.parameters():
        torch.nn.init.zeros_(parameter)
    torch.save({"d_in": 2, "state_dict": head.state_dict()}, tmp_path / "ffn.pt")
    args = arguments(tmp_path, "ffn")
    args[args.index("--ffn-checkpoint") + 1] = str(tmp_path / "ffn.pt")
    cli.main(["cut", *args, "--output-format", output_format])
    csv_expected = output_format in {"csv", "both"}
    fasta_expected = output_format in {"fasta", "both"}
    assert (tmp_path / "products_residue_predictions.csv").exists() == csv_expected
    assert (tmp_path / "products_sequence_summary.csv").exists() == csv_expected
    assert (tmp_path / "products_cleaved_products.fasta").exists() == fasta_expected
    if csv_expected:
        residues = pd.read_csv(tmp_path / "products_residue_predictions.csv")
        assert residues["residue_index_1based"].tolist() == [1, 2, 3, 4]
        assert residues["prob_selected"].tolist() == [0.5] * 4
        assert residues["is_cleavage"].tolist() == [0] * 4
        summary = pd.read_csv(tmp_path / "products_sequence_summary.csv")
        assert summary["n_products"].tolist() == [1]
    if fasta_expected:
        assert (tmp_path / "products_cleaved_products.fasta").read_text() == ">protein_cut[1:4]\nACDE\n"


def test_cut_positions_split_after_residue():
    assert cutting.cleave_sequence("ACDE", [2]) == [(1, 2, "AC"), (3, 4, "DE")]


def test_default_cutter_subfolder():
    args = cutting.parse_args([
        "--input-fasta", "input.fa", "--cutter-model", "owner/repo",
        "--output-prefix", "out",
    ])
    assert args.cutter_subfolder == "cutter"

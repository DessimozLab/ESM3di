import pytest
import pandas as pd
from pathlib import Path
from esm3di.inference import ESM3DiPredictor


@pytest.fixture(scope="module")
def predictor():
    """Initializes the ESM3DiPredictor engine once for the testing lifecycle."""
    # Points cleanly to your local checkpoint tracking weights
    checkpoint = Path(__file__).resolve().parents[1] / "checkpoints" / "hf_compatible"
    return ESM3DiPredictor(model_checkpoint_path=checkpoint)


@pytest.fixture
def sample_fasta(tmp_path):
    """Creates a temporary isolated test FASTA file."""
    fasta_file = tmp_path / "test_input.fasta"
    fasta_file.write_text(">seq_alpha\nMAEGEITTFTALTEKFNLPPGNYK\n>seq_beta\nMAEGEITTFTAL")
    return str(fasta_file)


def test_predict_single_and_batch(predictor):
    """Validates basic inference string generation, token mappings, and sequence constraints."""
    test_seq = "MAEGEITTFTAL"

    # 1. Test single string inference API
    single_out = predictor.predict(test_seq)
    assert isinstance(single_out, str)
    assert len(single_out) == len(test_seq), "Output length mismatch in single prediction."

    # 2. Test batch execution pipeline mapping
    batch_seqs = ["MAEGEITTFTAL", "TEKFNLPPGNYK"]
    batch_out = predictor.predict_batch(batch_seqs, batch_size=2)
    assert len(batch_out) == 2
    assert len(batch_out[0]) == len(batch_seqs[0])
    assert len(batch_out[1]) == len(batch_seqs[1])

    # 3. Alphabet constraint enforcement
    valid_3di_chars = set("ACDEFGHIKLMNPQRSTVWYX")
    for out_str in batch_out:
        assert set(out_str.upper()).issubset(valid_3di_chars), "Generated invalid structural vocabulary tokens."


def test_per_position_perplexity_tsv(predictor, sample_fasta, tmp_path):
    """Validates the mathematical properties of the Shannon entropy tracking engine."""
    output_tsv = tmp_path / "metrics.tsv"

    # 1. Execute the position-wise perplexity calculation engine
    predictor.output_per_position_perplexity(
        input_fasta_path=sample_fasta,
        output_tsv_path=str(output_tsv),
        batch_size=2
    )

    assert output_tsv.exists(), "Perplexity metric TSV was not written to disk."

    # 2. Read back data to inspect structural format boundaries
    df = pd.read_csv(output_tsv, sep='\t')

    # Expected structured headers check
    expected_columns = ["sequence_id", "position", "aa", "perplexity"]
    assert list(df.columns) == expected_columns, "TSV columns do not match production architecture standard."

    # Assert total coordinate rows match true unmasked residues across the batch (24 + 12 = 36)
    assert len(df) == 36, "Position indexing counts skipped real residues or included padding masks."

    # 3. Scientific Assertions: Boundary distribution constraints
    # (Based on your empirical metrics: min ~1.001, median ~4.05, max ~14.44)
    assert (df['perplexity'] >= 1.0).all(), "Detected negative or mathematically impossible perplexity values (< 1.0)."
    assert (df['perplexity'] < 20.0).all(), "Perplexity exploded past maximum uniform distribution bounds (> 20.0)."

    # Median confidence bounds assertion
    median_ppl = df['perplexity'].median()
    assert 2.0 < median_ppl < 8.0, f"Model distribution shifted anomalously. Current median: {median_ppl:.2f}"
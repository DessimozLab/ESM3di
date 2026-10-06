from esm3di.io import (
    fasta2foldseek,
    iter_fasta,
    read_fasta,
    resolve_output_path,
    write_fasta,
)


def test_read_and_write_fasta_normalize_sequences_and_wrap_lines(tmp_path):
    input_path = tmp_path / "input.fasta"
    output_path = tmp_path / "output.fasta"
    sequence = "acgt" * 25
    input_path.write_text(f">seq description\n{sequence}\n\n")

    records = read_fasta(input_path)
    write_fasta(records, output_path)

    assert records == [("seq description", sequence.upper())]
    output_lines = output_path.read_text().splitlines()
    assert output_lines[0] == ">seq description"
    assert output_lines[1] == "ACGT" * 20
    assert output_lines[2] == "ACGT" * 5


def test_iter_fasta_supports_handles_cleaning_and_full_headers(tmp_path):
    path = tmp_path / "input.fasta"
    path.write_text(">seq one\nac-d.*\n>seq2\nEF\n")

    with path.open() as handle:
        records = list(iter_fasta(handle, clean="unalign", full_name=True))

    assert records == [("seq one", "ACD"), ("seq2", "EF")]


def test_resolve_output_path_places_bare_filename_in_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_output_path("result.fasta") == tmp_path / "outputs" / "result.fasta"
    assert resolve_output_path("nested/result.fasta") == tmp_path / "nested" / "result.fasta"


def test_fasta2foldseek_accepts_pathlike_inputs(tmp_path):
    aa_path = tmp_path / "aa.fasta"
    tdi_path = tmp_path / "tdi.fasta"
    output = tmp_path / "database"
    aa_path.write_text(">seq description\nACDE\n")
    tdi_path.write_text(">seq description\nFGHI\n")

    fasta2foldseek(aa_path, tdi_path, str(output))

    assert output.read_bytes() == b"ACDE\n\x00"
    assert (tmp_path / "database_ss").read_bytes() == b"FGHI\n\x00"
    assert (tmp_path / "database_h").read_bytes() == b"seq description\n\x00"

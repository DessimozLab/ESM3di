from esm3di.io import read_fasta
from esm3di.preprocessing import (
    _count_sequences,
    _merge_fasta_outputs,
    _shard_fasta,
)


def test_shard_fasta_round_robin_and_merge_restores_order(tmp_path):
    input_path = tmp_path / "input.fasta"
    input_path.write_text(">a\nA\n>b\nBB\n>c\nCCC\n>d\nDDDD\n")

    shards = _shard_fasta(str(input_path), 2, str(tmp_path / "shards"))

    assert _count_sequences(str(input_path)) == 4
    assert read_fasta(shards[0][0]) == [("a", "A"), ("c", "CCC")]
    assert read_fasta(shards[1][0]) == [("b", "BB"), ("d", "DDDD")]

    outputs = []
    for index, (shard_path, _) in enumerate(shards):
        output_path = tmp_path / f"output_{index}.fasta"
        records = read_fasta(shard_path)
        output_path.write_text("".join(f">{header}\n{sequence.lower()}\n" for header, sequence in records))
        outputs.append((shard_path, str(output_path)))

    merged_path = tmp_path / "merged.fasta"
    _merge_fasta_outputs(outputs, str(merged_path), ["a", "b", "c", "d"])

    assert read_fasta(merged_path) == [
        ("a", "a"),
        ("b", "bb"),
        ("c", "ccc"),
        ("d", "dddd"),
    ]

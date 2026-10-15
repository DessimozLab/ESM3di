import os
from typing import List, Tuple
from .io import iter_fasta, read_fasta, write_fasta






def _count_sequences(fasta_path: str) -> int:
    """Stream-based parsing for memory efficiency."""
    return sum(1 for _ in iter_fasta(fasta_path, full_name=True))


def _shard_fasta(input_fasta: str, num_shards: int, temp_dir: str,
                 sort_by_length: bool = False) -> List[Tuple[str, List[str]]]:
    """Distributes sequences using round-robin for multi-GPU inference."""
    os.makedirs(temp_dir, exist_ok=True)
    shards = [[] for _ in range(num_shards)]

    records = read_fasta(input_fasta)
    if sort_by_length:
        records.sort(key=lambda record: len(record[1]))

    for i, record in enumerate(records):
        shards[i % num_shards].append(record)

    result = []
    for gpu_id in range(num_shards):
        shard_path = os.path.join(temp_dir, f"shard_{gpu_id}_aa.fasta")
        write_fasta(shards[gpu_id], shard_path)
        result.append((shard_path, [h for h, _ in shards[gpu_id]]))
    return result


def _merge_fasta_outputs(shard_outputs: List[Tuple[str, str]], output_fasta: str, original_order: List[str]):
    """Merges 3Di prediction outputs back into original sequence order."""
    all_sequences = {}
    for _, shard_3di_path in shard_outputs:
        all_sequences.update(dict(read_fasta(shard_3di_path)))

    write_fasta([(h, all_sequences[h]) for h in original_order], output_fasta)
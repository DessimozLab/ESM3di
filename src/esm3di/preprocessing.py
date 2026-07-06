import os
import torch
from typing import Optional, List, Tuple
from torch.utils.data import Dataset
from .io import read_fasta, write_fasta


class Seq3DiDataset(Dataset):
    """Holds (amino_acid_sequence, 3Di_label_sequence, plddt_bins, aux_bins...) tuples."""

    def __init__(self, aa_fasta: str, three_di_fasta: str, mask_label_chars: str = "",
                 plddt_bins_fasta: str = None, aux_fastas: Optional[dict] = None):
        aa_records = read_fasta(aa_fasta)
        lab_records = read_fasta(three_di_fasta)

        self.has_plddt = plddt_bins_fasta is not None
        plddt_records = read_fasta(plddt_bins_fasta) if self.has_plddt else None

        self.aux_track_names = list(aux_fastas.keys()) if aux_fastas else []
        aux_records_by_name = {k: read_fasta(v) for k, v in (aux_fastas or {}).items()}

        self.items = []
        all_chars = set()
        self.mask_label_chars = set() if self.has_plddt else set(mask_label_chars)

        for idx, ((h_aa, seq_aa), (h_lab, seq_lab)) in enumerate(zip(aa_records, lab_records)):
            plddt_seq = plddt_records[idx][1] if self.has_plddt else None
            aux_seqs = {k: aux_records_by_name[k][idx][1] for k in self.aux_track_names}

            self.items.append((h_aa, seq_aa, seq_lab, plddt_seq, aux_seqs))
            all_chars.update(seq_lab)

        self.label_vocab = sorted(ch for ch in all_chars if ch not in self.mask_label_chars)
        self.char2idx = {c: i for i, c in enumerate(self.label_vocab)}

    def __len__(self): return len(self.items)

    def __getitem__(self, idx): return self.items[idx]


def make_collate_fn(tokenizer, char2idx, mask_label_chars: str = "",
                    include_plddt: bool = False, max_seq_length: int = None,
                    aux_track_names: Optional[list] = None):
    """Tokenizes AA sequences with HF tokenizer and aligns targets."""
    mask_set = set() if include_plddt else set(mask_label_chars)
    _aux_track_names = list(aux_track_names) if aux_track_names else []
    _char_to_bin = {ch: i for i, ch in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")}

    def collate(batch):
        # Unpack batch (accounting for missing optional elements)
        headers, aa_seqs, label_seqs = [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]
        plddt_seqs = [b[3] if len(b) > 3 else None for b in batch]
        aux_seqs_list = [b[4] if len(b) > 4 else {} for b in batch]

        enc = tokenizer(list(aa_seqs), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_seq_length, add_special_tokens=True, return_special_tokens_mask=True)

        input_ids, attention_mask, special_mask = enc["input_ids"], enc["attention_mask"], enc["special_tokens_mask"]
        batch_size, max_len = input_ids.shape

        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        plddt_bins = torch.zeros((batch_size, max_len), dtype=torch.long) if include_plddt else None
        aux_bins = {t: torch.full((batch_size, max_len), -100, dtype=torch.long) for t in _aux_track_names}

        for i, lab_seq in enumerate(label_seqs):
            k = 0
            for j in range(max_len):
                if special_mask[i, j] == 1:
                    continue  # Remains -100
                if k < len(lab_seq):
                    ch = lab_seq[k]
                    if ch not in mask_set and ch in char2idx:
                        labels[i, j] = char2idx[ch]

                    if include_plddt and plddt_seqs[i]:
                        plddt_bins[i, j] = int(plddt_seqs[i][k])

                    for track in _aux_track_names:
                        if k < len(aux_seqs_list[i].get(track, "")):
                            aux_bins[track][i, j] = _char_to_bin[aux_seqs_list[i][track][k]]
                    k += 1

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if plddt_bins is not None: out["plddt_bins"] = plddt_bins
        if aux_bins: out["aux_bins"] = aux_bins
        return out

    return collate


def _count_sequences(fasta_path: str) -> int:
    """Stream-based parsing for memory efficiency."""
    from Bio import SeqIO
    return sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))


def _shard_fasta(input_fasta: str, num_shards: int, temp_dir: str) -> List[Tuple[str, List[str]]]:
    """Distributes sequences using round-robin for multi-GPU inference."""
    from Bio import SeqIO
    shards = [[] for _ in range(num_shards)]

    for i, record in enumerate(SeqIO.parse(input_fasta, "fasta")):
        shards[i % num_shards].append((record.id, str(record.seq)))

    result = []
    for gpu_id in range(num_shards):
        shard_path = os.path.join(temp_dir, f"shard_{gpu_id}_aa.fasta")
        write_fasta(shards[gpu_id], shard_path)
        result.append((shard_path, [h for h, _ in shards[gpu_id]]))
    return result


def _merge_fasta_outputs(shard_outputs: List[Tuple[str, str]], output_fasta: str, original_order: List[str]):
    """Merges 3Di prediction outputs back into original sequence order."""
    from Bio import SeqIO
    all_sequences = {}
    for _, shard_3di_path in shard_outputs:
        for record in SeqIO.parse(shard_3di_path, "fasta"):
            all_sequences[record.id] = str(record.seq)

    write_fasta([(h, all_sequences[h]) for h in original_order], output_fasta)
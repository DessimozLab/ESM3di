"""Full-length residue inference with bounded, overlapping model inputs."""

from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch


def window_ranges(length: int, window: int, overlap: int) -> list[Tuple[int, int]]:
    if length <= window:
        return [(0, length)]
    starts = list(range(0, length - window, window - overlap)) + [length - window]
    return [(start, start + window) for start in sorted(set(starts))]


def residue_window_size(tokenizer) -> int:
    # Tokenizers without a meaningful limit use a huge sentinel value.
    tokens = min(1024, int(tokenizer.model_max_length))
    window = tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if window < 1:
        raise ValueError("Tokenizer context must leave room for at least one residue")
    return window


@torch.no_grad()
def windowed_outputs(
    sequences: Sequence[str],
    tokenizer,
    device,
    batch_size: int,
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    truncate: bool = False,
) -> Iterator[torch.Tensor]:
    """Yield CPU residue tensors, averaging overlapping windows in input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    window = residue_window_size(tokenizer)
    overlap = min(128, window // 2)
    for offset in range(0, len(sequences), batch_size):
        batch = sequences[offset:offset + batch_size]
        lengths = [min(len(seq), window) if truncate else len(seq) for seq in batch]
        jobs = [
            (idx, start, end)
            for idx, length in enumerate(lengths)
            for start, end in window_ranges(length, window, overlap)
            if end > start
        ]
        sums: List[Optional[torch.Tensor]] = [None] * len(batch)
        counts = [torch.zeros(length, 1) for length in lengths]
        for start in range(0, len(jobs), batch_size):
            chunk_jobs = jobs[start:start + batch_size]
            chunks = [batch[idx][left:right] for idx, left, right in chunk_jobs]
            enc = tokenizer(
                chunks, return_tensors="pt", padding=True, truncation=False,
                add_special_tokens=True, return_special_tokens_mask=True,
            )
            if enc["input_ids"].shape[1] > window + tokenizer.num_special_tokens_to_add(pair=False):
                raise RuntimeError("Tokenizer exceeded the model window; expected one token per residue")
            values = forward(
                enc["input_ids"].to(device), enc["attention_mask"].to(device),
            ).detach().float().cpu()
            masks = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
            for row, (idx, left, right) in enumerate(chunk_jobs):
                residues = values[row, masks[row]]
                if residues.shape[0] != right - left:
                    raise RuntimeError(
                        f"Tokenizer produced {residues.shape[0]} residue tokens for a "
                        f"{right - left}-residue window; refusing incomplete predictions"
                    )
                total = sums[idx]
                if total is None:
                    total = torch.zeros(lengths[idx], residues.shape[-1])
                    sums[idx] = total
                total[left:right] += residues
                counts[idx][left:right] += 1
        for total, count, length in zip(sums, counts, lengths):
            if length == 0:
                yield torch.empty(0, 0)
            elif total is None or torch.any(count == 0):
                raise RuntimeError("Windowed inference left residues without predictions")
            else:
                yield total / count

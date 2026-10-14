"""Predict 3Di sequences from amino-acid FASTA with a trained ESM3Di checkpoint.

Works with every ``.pt`` checkpoint written by ``esm3di.train_3di`` and with
the older ``esmretrain.py`` checkpoints (LoRA + linear/CNN head).

Long sequences are handled with overlapping windows (probabilities are averaged
in the overlaps), so no residue is ever dropped. Sequences are processed in
length-sorted batches and written back in the input order with their full
original headers.

Example::

    python -m esm3di.predict \
        --checkpoint runs/3di/best.pt \
        --input proteins.fasta \
        --output proteins_3di.fasta \
        --confidence-output proteins_3di_conf.fasta

Optionally build a FoldSeek database directly from the predictions::

    python -m esm3di.predict --checkpoint best.pt --input proteins.fasta \
        --output proteins_3di.fasta --foldseek-db dbs/proteins
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from .checkpoints import load_3di_checkpoint

Record = Tuple[str, str]  # (full header without '>', sequence)


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def read_fasta_records(path: str) -> List[Record]:
    """Read FASTA (optionally gzipped) keeping the full header line."""
    opener = gzip.open if path.endswith(".gz") else open
    records: List[Record] = []
    header, chunks = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def write_fasta_records(records: Iterable[Record], path: str, width: int = 0) -> None:
    """Write records; ``width=0`` writes each sequence on one line."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            if width and width > 0:
                for i in range(0, len(seq), width):
                    fh.write(seq[i : i + width] + "\n")
            else:
                fh.write(seq + "\n")
    os.replace(tmp, path)


def clean_sequence(seq: str) -> str:
    """Uppercase and drop gap/stop characters; unknown residues stay (tokenizer maps them to X/unk)."""
    return seq.upper().replace("-", "").replace(".", "").replace("*", "")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Holds a loaded 3Di tagger and turns AA sequences into 3Di strings."""

    def __init__(
        self,
        checkpoint: str,
        device: Optional[str] = None,
        hf_revision: Optional[str] = None,
        amp: bool = True,
        window: Optional[int] = None,
        overlap: int = 128,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.esm, ckpt = load_3di_checkpoint(checkpoint, device=str(self.device), hf_revision=hf_revision)
        self.model = self.esm.model
        self.tokenizer = self.esm.tokenizer
        self.label_vocab: List[str] = list(ckpt["label_vocab"])
        args = ckpt.get("args") or {}
        # Residues per window = training length minus CLS/EOS.
        max_tokens = window + 2 if window else int(args.get("max_seq_length") or 1024)
        self.window = max(8, max_tokens - 2)
        self.overlap = max(0, min(overlap, self.window // 2))
        self.amp = amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.checkpoint_epoch = ckpt.get("epoch")

    # -- windowing -----------------------------------------------------------
    def _windows(self, length: int) -> List[Tuple[int, int]]:
        if length <= self.window:
            return [(0, length)]
        step = self.window - self.overlap
        starts = list(range(0, length - self.window, step)) + [length - self.window]
        return [(s, s + self.window) for s in sorted(set(starts))]

    @torch.no_grad()
    def _probs(self, chunks: Sequence[str]) -> List[torch.Tensor]:
        """Per-residue softmax probabilities (L_i, n_labels) for each chunk."""
        enc = self.tokenizer(
            list(chunks),
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        special = enc["special_tokens_mask"].bool()
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        ctx = torch.autocast(self.device.type, dtype=self.amp_dtype) if self.amp else nullcontext()
        with ctx:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        out = []
        for i, chunk in enumerate(chunks):
            positions = (~special[i] & enc["attention_mask"][i].bool()).nonzero(as_tuple=True)[0]
            if len(positions) != len(chunk):
                raise RuntimeError(
                    f"Tokenizer produced {len(positions)} residue tokens for a {len(chunk)}-residue chunk; "
                    "the input probably contains characters the tokenizer merges or splits."
                )
            out.append(probs[i, positions])
        return out

    def predict(
        self, records: Sequence[Record], batch_size: int = 8, max_batch_residues: int = 16384, progress: bool = True
    ) -> Iterator[Tuple[int, str, str]]:
        """Yield ``(input_index, three_di, confidence_digits)`` in length-sorted order."""
        # Expand into windows, sort by length so padding is minimal.
        jobs = []  # (record_index, start, end)
        seqs = [clean_sequence(s) for _, s in records]
        for idx, seq in enumerate(seqs):
            for start, end in self._windows(len(seq)):
                jobs.append((idx, start, end))
        jobs.sort(key=lambda j: j[2] - j[1], reverse=True)

        pending = {}  # record_index -> [accum_probs, counts, windows_left]
        for idx, seq in enumerate(seqs):
            pending[idx] = [None, None, len(self._windows(len(seq)))]

        t0, done = time.time(), 0
        i = 0
        while i < len(jobs):
            longest = jobs[i][2] - jobs[i][1]
            n = max(1, min(batch_size, max_batch_residues // max(longest, 1)))
            batch = jobs[i : i + n]
            i += n
            chunk_probs = self._probs([seqs[r][s:e] for r, s, e in batch]) if longest > 0 else []
            for (r, s, e), p in zip(batch, chunk_probs):
                state = pending[r]
                if state[0] is None:
                    L = len(seqs[r])
                    state[0] = torch.zeros(L, len(self.label_vocab))
                    state[1] = torch.zeros(L, 1)
                state[0][s:e] += p
                state[1][s:e] += 1
                state[2] -= 1
                if state[2] == 0:
                    probs = state[0] / state[1].clamp_min(1)
                    conf, pred = probs.max(dim=-1)
                    three_di = "".join(self.label_vocab[k] for k in pred.tolist())
                    digits = "".join(str(min(9, int(c * 10))) for c in conf.tolist())
                    del pending[r]
                    done += 1
                    yield r, three_di, digits
            if progress and (done % 200 < n or i >= len(jobs)):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"\r  predicted {done}/{len(records)} sequences ({rate:.1f}/s)", end="", file=sys.stderr)
        # Zero-length sequences produce no windows.
        for r in list(pending):
            yield r, "", ""
        if progress:
            print(file=sys.stderr)


def predict_records(
    records: Sequence[Record],
    checkpoint: str,
    device: Optional[str] = None,
    hf_revision: Optional[str] = None,
    batch_size: int = 8,
    max_batch_residues: int = 16384,
    amp: bool = True,
    window: Optional[int] = None,
    overlap: int = 128,
    progress: bool = True,
) -> List[Tuple[str, str, str]]:
    """Return ``[(header, three_di, confidence)]`` in input order."""
    predictor = Predictor(checkpoint, device=device, hf_revision=hf_revision, amp=amp, window=window, overlap=overlap)
    results: List[Optional[Tuple[str, str]]] = [None] * len(records)
    for idx, three_di, conf in predictor.predict(records, batch_size, max_batch_residues, progress):
        results[idx] = (three_di, conf)
    return [(records[i][0], *results[i]) for i in range(len(records))]


# ---------------------------------------------------------------------------
# Multi-GPU (one process per GPU)
# ---------------------------------------------------------------------------

def _worker(gpu: int, shard_path: str, out_path: str, kwargs: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    records = read_fasta_records(shard_path)
    results = predict_records(records, device="cuda", progress=False, **kwargs)
    torch.save(results, out_path)


def predict_multi_gpu(records: Sequence[Record], gpus: Sequence[int], **kwargs) -> List[Tuple[str, str, str]]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="esm3di_predict_") as tmp:
        procs, outputs = [], []
        # Interleave by length so shards have similar work.
        order = sorted(range(len(records)), key=lambda i: len(records[i][1]), reverse=True)
        shards = [order[k :: len(gpus)] for k in range(len(gpus))]
        for k, (gpu, shard) in enumerate(zip(gpus, shards)):
            shard_path = os.path.join(tmp, f"shard{k}.fasta")
            out_path = os.path.join(tmp, f"shard{k}.pt")
            # Write headers as indices so the merge is unambiguous.
            write_fasta_records(((str(i), records[i][1]) for i in shard), shard_path)
            p = ctx.Process(target=_worker, args=(gpu, shard_path, out_path, kwargs))
            p.start()
            procs.append(p)
            outputs.append(out_path)
        for p in procs:
            p.join()
        failed = [gpu for gpu, p in zip(gpus, procs) if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Prediction worker(s) on GPU(s) {failed} failed; see the messages above.")
        merged: List[Optional[Tuple[str, str, str]]] = [None] * len(records)
        for out_path in outputs:
            for idx, three_di, conf in torch.load(out_path, weights_only=False):
                i = int(idx)
                merged[i] = (records[i][0], three_di, conf)
    return merged  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_predict_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--checkpoint", "--model-ckpt", dest="checkpoint", required=True, help="Trained 3Di .pt checkpoint.")
    p.add_argument("--input", "-i", "--aa-fasta", dest="input", required=True, help="Amino-acid FASTA (.gz ok).")
    p.add_argument("--output", "-o", required=True, help="Output 3Di FASTA.")
    p.add_argument("--confidence-output", help="Optional FASTA of per-residue confidence digits (max prob * 10, 0-9).")
    p.add_argument("--lowercase", action="store_true", help="Write 3Di in lowercase (default uppercase, as FoldSeek).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batch-residues", type=int, default=16384, help="Cap on batch_size * longest window.")
    p.add_argument("--window", type=int, help="Residues per window for long sequences (default: training max_seq_length - 2).")
    p.add_argument("--overlap", type=int, default=128, help="Residue overlap between windows.")
    p.add_argument("--device", help="cuda, cuda:1, cpu (default: cuda if available).")
    p.add_argument("--gpus", help="Comma-separated GPU ids for multi-GPU inference, e.g. '0,1'.")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision on GPU.")
    p.add_argument("--hf-revision", help="Override the pinned ESM++ code revision.")
    p.add_argument("--foldseek-db", help="Also write a FoldSeek database at this path prefix.")
    return p


def run_predict(ns: argparse.Namespace) -> List[Tuple[str, str, str]]:
    records = read_fasta_records(ns.input)
    if not records:
        raise SystemExit(f"No sequences found in {ns.input}")
    print(f"Read {len(records)} sequences from {ns.input}")
    kwargs = dict(
        checkpoint=ns.checkpoint,
        hf_revision=ns.hf_revision,
        batch_size=ns.batch_size,
        max_batch_residues=ns.max_batch_residues,
        amp=not ns.no_amp,
        window=ns.window,
        overlap=ns.overlap,
    )
    t0 = time.time()
    gpus = [int(g) for g in ns.gpus.split(",")] if ns.gpus else []
    if len(gpus) > 1:
        results = predict_multi_gpu(records, gpus, **kwargs)
    else:
        device = ns.device or (f"cuda:{gpus[0]}" if gpus else None)
        results = predict_records(records, device=device, **kwargs)

    three_di = [(h, s.lower() if ns.lowercase else s) for h, s, _ in results]
    write_fasta_records(three_di, ns.output)
    print(f"Wrote {len(results)} 3Di sequences to {ns.output} in {time.time() - t0:.1f}s")
    if ns.confidence_output:
        write_fasta_records(((h, c) for h, _, c in results), ns.confidence_output)
        print(f"Wrote confidences to {ns.confidence_output}")
    if ns.foldseek_db:
        from .foldseek_db import write_foldseek_db

        aa = [(h, clean_sequence(s)) for h, s in records]
        write_foldseek_db(aa, [(h, s.upper()) for h, s, _ in results], ns.foldseek_db)
        print(f"Wrote FoldSeek database {ns.foldseek_db}")
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_predict_args(parser)
    run_predict(parser.parse_args(argv))


if __name__ == "__main__":
    main()

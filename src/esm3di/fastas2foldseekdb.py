"""Build a FoldSeek database from an amino-acid FASTA using predicted 3Di.

This is the backwards-compatible entry point (``esm3di-makefoldseekdb``); it
runs :mod:`esm3di.predict` and :mod:`esm3di.foldseek_db`. The new CLI exposes
the same thing as ``esm3di predict --foldseek-db``.

Examples::

    # Predict 3Di with a trained checkpoint and write the DB
    python -m esm3di.fastas2foldseekdb --aa-fasta proteins.fasta \
        --model-ckpt runs/3di/best.pt --output-db dbs/proteins

    # Use pre-computed 3Di
    python -m esm3di.fastas2foldseekdb --aa-fasta proteins.fasta \
        --three-di-fasta proteins_3di.fasta --skip-inference --output-db dbs/proteins
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from .foldseek_db import fasta2foldseek, write_foldseek_db
from .predict import clean_sequence, predict_multi_gpu, predict_records, read_fasta_records, write_fasta_records

__all__ = ["main", "fasta2foldseek", "write_foldseek_db", "create_foldseek_db_from_fastas"]


def create_foldseek_db_from_fastas(aa_fasta: str, three_di_fasta: str, output_db: str, **_ignored) -> bool:
    """Compatibility wrapper (older signature accepted ``foldseek_bin`` etc.)."""
    fasta2foldseek(aa_fasta, three_di_fasta, output_db)
    return True


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aa-fasta", required=True, help="Input amino-acid FASTA.")
    p.add_argument("--three-di-fasta", help="Pre-computed 3Di FASTA (with --skip-inference).")
    p.add_argument("--model-ckpt", help="Trained 3Di .pt checkpoint.")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--device")
    p.add_argument("--num-gpus", type=int, help="Use GPUs 0..N-1 (default: 1 GPU or CPU).")
    p.add_argument("--revision", "--model-revision", dest="revision", help="Override the pinned ESM++ code revision.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-db", required=True, help="Output FoldSeek DB path prefix.")
    p.add_argument("--keep-fastas", action="store_true", help="Also write <output-db>_aa.fasta and <output-db>_3di.fasta.")
    p.add_argument("--output-aa-fasta")
    p.add_argument("--output-3di-fasta")
    p.add_argument("--output-confidence-fasta")
    p.add_argument("--foldseek-bin", help="Unused (the DB is written directly); kept for compatibility.")
    a = p.parse_args(argv)

    if not os.path.exists(a.aa_fasta):
        p.error(f"Input file not found: {a.aa_fasta}")
    aa = [(h, clean_sequence(s)) for h, s in read_fasta_records(a.aa_fasta)]

    if a.skip_inference:
        if not a.three_di_fasta:
            p.error("--three-di-fasta is required with --skip-inference")
        tdi = [(h, s.upper()) for h, s in read_fasta_records(a.three_di_fasta)]
        conf = None
    else:
        if not a.model_ckpt:
            p.error("--model-ckpt is required unless --skip-inference is set")
        kwargs = dict(checkpoint=a.model_ckpt, hf_revision=a.revision, batch_size=a.batch_size)
        if a.num_gpus and a.num_gpus > 1:
            results = predict_multi_gpu(aa, list(range(a.num_gpus)), **kwargs)
        else:
            results = predict_records(aa, device=a.device, **kwargs)
        tdi = [(h, s) for h, s, _ in results]
        conf = [(h, c) for h, _, c in results]

    write_foldseek_db(aa, tdi, a.output_db)
    print(f"Wrote FoldSeek database {a.output_db} ({len(aa)} entries)")

    if a.keep_fastas or a.output_aa_fasta:
        write_fasta_records(aa, a.output_aa_fasta or f"{a.output_db}_aa.fasta")
    if (a.keep_fastas or a.output_3di_fasta) and not a.skip_inference:
        write_fasta_records(tdi, a.output_3di_fasta or f"{a.output_db}_3di.fasta")
    if a.output_confidence_fasta:
        if conf is None:
            print("--output-confidence-fasta ignored with --skip-inference", file=sys.stderr)
        else:
            write_fasta_records(conf, a.output_confidence_fasta)


if __name__ == "__main__":
    main()

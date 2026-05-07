#!/usr/bin/env python
"""
Extract AA and 3Di FASTA files from a FoldSeek database.

This is a small utility around the FoldSeek DB files created by
`foldseek createdb`, mirroring the logic in build_trainingset.read_dbfiles3di.
"""

import argparse
from typing import Dict, Tuple

from tqdm import tqdm


def read_dbfiles3di(aadb: str, three_di_db: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Read FoldSeek DB files and return 3Di/AA sequence mappings."""
    three_di_seq = [line.strip().replace("\x00", "") for line in open(three_di_db)]
    lookup = aadb + ".lookup"
    ids = [
        line.split()[1].strip().replace(".pdb", "").split("/")[-1]
        for line in open(lookup)
    ]
    aas = [line.strip().replace("\x00", "") for line in open(aadb)]
    mapper3di = dict(zip(ids, three_di_seq))
    mapperaa = dict(zip(ids, aas))
    return mapper3di, mapperaa


def mapper2fasta(mapper3di: Dict[str, str], mapperaa: Dict[str, str], output_prefix: str):
    """Write AA and 3Di sequences to FASTA files."""
    aa_fasta_path = f"{output_prefix}_aa.fasta"
    three_di_fasta_path = f"{output_prefix}_3di.fasta"

    with open(aa_fasta_path, "w") as aa_fasta:
        for seq_id, seq in tqdm(mapperaa.items(), desc="Writing AA sequences", unit="seq"):
            aa_fasta.write(f">{seq_id}\n{seq}\n")

    with open(three_di_fasta_path, "w") as three_di_fasta:
        for seq_id, seq in tqdm(mapper3di.items(), desc="Writing 3Di sequences", unit="seq"):
            three_di_fasta.write(f">{seq_id}\n{seq}\n")

    return aa_fasta_path, three_di_fasta_path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract FASTA files from a FoldSeek database.")
    parser.add_argument(
        "--db-prefix",
        required=True,
        help="FoldSeek DB prefix (path used for createdb output).",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output FASTA files.",
    )
    parser.add_argument(
        "--three-di-db",
        default=None,
        help="Optional 3Di DB path. Defaults to <db-prefix>_ss.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    three_di_db = args.three_di_db or f"{args.db_prefix}_ss"
    mapper3di, mapperaa = read_dbfiles3di(args.db_prefix, three_di_db)
    aa_fasta, three_di_fasta = mapper2fasta(mapper3di, mapperaa, args.output_prefix)
    print(f"Wrote AA FASTA: {aa_fasta}")
    print(f"Wrote 3Di FASTA: {three_di_fasta}")


if __name__ == "__main__":
    main()

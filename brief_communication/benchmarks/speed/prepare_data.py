"""
Downloads Swiss-Prot viral proteins, filters by sequence length, shuffles reproducibly,
and outputs nested FASTA files (subset_10.fasta, subset_100.fasta, etc.) for speed benchmarking.

NOTE: 
If your brief communication positions ems3di as a general-purpose tool: a large seuqence length range is appropriate for the dataset
It proves robust performance across diverse viral protein sizes. This is what the current dataset is designed for (default: --min-len 50 --max-len 1000).

If the paper more explicitly focuses on structural phylogenetics: Re-run prepare_data.py with a tight length bound (e.g., --min-len 380 --max-len 420).
"""

import argparse
import random
import sys
import urllib.request
from pathlib import Path

# UniProt REST API endpoint for reviewed (Swiss-Prot) viral sequences (~17k sequences)
UNIPROT_VIRAL_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "format=fasta&query=%28taxonomy_id%3A10239%29+AND+%28reviewed%3Atrue%29"
)

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_uniprot_viral(output_path: Path) -> None:
    """Downloads viral FASTA dataset from UniProt if not present locally."""
    if output_path.exists():
        print(f"[+] Found cached raw fasta at: {output_path}")
        return

    print(f"[+] Downloading Swiss-Prot viral dataset from UniProt...")
    print(f"    URL: {UNIPROT_VIRAL_URL}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(UNIPROT_VIRAL_URL, output_path)
        print(f"[+] Download complete: {output_path}")
    except Exception as e:
        print(f"[!] Error downloading data: {e}", file=sys.stderr)
        sys.exit(1)


def parse_fasta(fasta_path: Path):
    """Yields (header, sequence) tuples from a FASTA file."""
    header = None
    seq_chunks = []
    
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def is_valid_sequence(seq: str, min_len: int, max_len: int) -> bool:
    """Filters sequences by length bounds and standard amino acid composition."""
    if not (min_len <= len(seq) <= max_len):
        return False
    # Exclude sequences with non-standard/ambiguous AA characters (X, B, Z, U, O)
    if not set(seq.upper()).issubset(VALID_AMINO_ACIDS):
        return False
    return True


def create_nested_subsets(
    raw_fasta: Path,
    output_dir: Path,
    subset_sizes: list[int],
    min_len: int = 50,
    max_len: int = 1000,
    seed: int = 42,
) -> None:
    """Filters, shuffles, and exports nested subset FASTA files."""
    print(f"[+] Parsing and filtering sequences ({min_len} <= length <= {max_len})...")
    
    filtered_records = []
    for header, seq in parse_fasta(raw_fasta):
        if is_valid_sequence(seq, min_len, max_len):
            filtered_records.append((header, seq))

    print(f"[+] Total valid sequences passing filters: {len(filtered_records)}")

    # Shuffle once with a fixed seed to establish nested ordering
    random.seed(seed)
    random.shuffle(filtered_records)

    output_dir.mkdir(parents=True, exist_ok=True)

    for n in subset_sizes:
        if n > len(filtered_records):
            print(
                f"[!] Warning: Requested subset size {n} exceeds total available sequences "
                f"({len(filtered_records)}). Truncating to max available."
            )
            n_actual = len(filtered_records)
        else:
            n_actual = n

        subset = filtered_records[:n_actual]
        out_file = output_dir / f"subset_{n}.fasta"

        # Calculate dataset statistics
        lengths = [len(seq) for _, seq in subset]
        mean_len = sum(lengths) / len(lengths)

        with open(out_file, "w", encoding="utf-8") as f:
            for header, seq in subset:
                f.write(f"{header}\n{seq}\n")

        print(
            f"[+] Written {out_file.name}: {len(subset)} sequences | "
            f"Mean length: {mean_len:.1f} aa (Min: {min(lengths)}, Max: {max(lengths)})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate nested viral FASTA subsets for benchmarking."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("brief_communication/benchmarks/speed/data"),
        help="Directory to store raw fasta and output subsets.",
    )
    parser.add_argument(
        "--subsets",
        type=int,
        nargs="+",
        default=[10, 100, 1000, 10000],
        help="Subset sizes to generate (default: 10 100 1000 10000).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=50,
        help="Minimum sequence length limit (default: 50).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1000,
        help="Maximum sequence length limit (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )

    args = parser.parse_args()

    raw_fasta_path = args.data_dir / "raw_viral_swissprot.fasta"
    
    # Step 1: Download raw viral data
    fetch_uniprot_viral(raw_fasta_path)

    # Step 2: Generate nested subsets
    create_nested_subsets(
        raw_fasta=raw_fasta_path,
        output_dir=args.data_dir,
        subset_sizes=sorted(args.subsets),
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
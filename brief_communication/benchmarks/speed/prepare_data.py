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
from Bio import SeqIO

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
    """Filters, shuffles, length-sorts, and exports nested subset FASTA files."""
    print(f"[+] Parsing and filtering sequences ({min_len} <= length <= {max_len})...")
    
    filtered_records = []
    for header, seq in parse_fasta(raw_fasta):
        if is_valid_sequence(seq, min_len, max_len):
            filtered_records.append((header, seq))

    print(f"[+] Total valid sequences passing filters: {len(filtered_records)}")

    # 1. Shuffle once with a fixed seed to establish reproducible random sampling
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

        # 2. Slice the random subset
        subset = filtered_records[:n_actual]

        # 3. Sort subset by sequence length (shortest to longest)
        subset.sort(key=lambda x: len(x[1]))

        out_file = output_dir / f"subset_{n}.fasta"

        # Calculate dataset statistics
        lengths = [len(seq) for _, seq in subset]
        mean_len = sum(lengths) / len(lengths)
        std_len = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5
        median_len = sorted(lengths)[len(lengths) // 2]
        Q1 = sorted(lengths)[len(lengths) // 4]
        Q3 = sorted(lengths)[3 * len(lengths) // 4]



        # 4. Write length-sorted sequences to file
        with open(out_file, "w", encoding="utf-8") as f:
            for header, seq in subset:
                f.write(f"{header}\n{seq}\n")

        print(
            f"[+] Written {out_file.name}: {len(subset)} sequences | "
            f"Sorted length range: {min(lengths)} -> {max(lengths)} aa | Mean: {mean_len:.1f} ± {std_len:.1f} aa | Median: {median_len} aa | Q1: {Q1} aa | Q3: {Q3} aa"
        )


def create_length_gradient_subset(
    raw_fasta: Path,
    output_fasta: Path,
    target_lengths: list[int] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    tolerance: int = 1,
):
    """
    Selects 1 representative protein for each target sequence length.
    """
    records = list(SeqIO.parse(raw_fasta, "fasta"))
    selected = []

    for target in target_lengths:
        # Find sequences within tolerance window of target length
        matches = [r for r in records if abs(len(r.seq) - target) <= tolerance]
        if not matches:
            # Fallback: pick the closest sequence in the dataset
            matches = [min(records, key=lambda r: abs(len(r.seq) - target))]
        
        # Select the first match
        chosen = matches[0]
        # Modify ID to clearly indicate length in logs/plots
        chosen.id = f"len_{len(chosen.seq)}aa_{chosen.id}"
        selected.append(chosen)

    SeqIO.write(selected, output_fasta, "fasta")
    print(f"[+] Created length gradient FASTA with {len(selected)} sequences at: {output_fasta}")


def create_length_binned_subsets(
    raw_fasta: Path,
    output_dir: Path,
    target_lengths: list[int] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    seqs_per_bin: int = 20,
    tolerance: int = 5,
):
    """
    Creates FASTA files for each target length step, containing `seqs_per_bin` 
    sequences around that length to amortize fixed script/IO overhead.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(SeqIO.parse(raw_fasta, "fasta"))

    for target in target_lengths:
        # Filter sequences within length tolerance
        matches = [r for r in records if abs(len(r.seq) - target) <= tolerance]
        
        # Fallback if window is too narrow
        if len(matches) < seqs_per_bin:
            matches = sorted(records, key=lambda r: abs(len(r.seq) - target))
        
        selected = matches[:seqs_per_bin]
        bin_fasta = output_dir / f"bin_L{target}.fasta"
        SeqIO.write(selected, bin_fasta, "fasta")
        
        avg_len = sum(len(r.seq) for r in selected) / len(selected)
        std_len = (sum((len(r.seq) - avg_len) ** 2 for r in selected) / len(selected)) ** 0.5
        print(f"[+] Bin L={target}aa: wrote {len(selected)} seqs (avg length: {avg_len:.1f} aa, std: {std_len:.1f} aa) -> {bin_fasta.name}")


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
        default=[1, 5, 10, 100, 500, 1000, 5000, 10000],
        help="Subset sizes to generate (default: [1, 5, 10, 100, 500, 1000, 5000, 10000]).",
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
    parser.add_argument(
        "--create-length-gradient",
        action="store_true",
        help="Create a length gradient subset FASTA (1 sequence per target length).",
    )
    parser.add_argument(
        "--create-length-bins",
        action="store_true",
        help="Create length-binned FASTA files (multiple sequences per target length).",
    )


    args = parser.parse_args()

    raw_fasta_path = args.data_dir / "raw_viral_swissprot.fasta"
    
    # Download raw viral data
    fetch_uniprot_viral(raw_fasta_path)

    # Create a length gradient subset
    if args.create_length_gradient:
        create_length_gradient_subset(
            raw_fasta=raw_fasta_path,
            output_fasta=args.data_dir / "subset_length_gradient.fasta",
            #target_lengths=sorted(args.subsets),
            tolerance=1
        )
    elif args.create_length_bins:
        create_length_binned_subsets(
            raw_fasta=raw_fasta_path,
            output_dir=args.data_dir / "length_bins",
            #target_lengths=sorted(args.subsets),
            seqs_per_bin=32,
            tolerance=2
        )
    else:
        # Generate nested subsets
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
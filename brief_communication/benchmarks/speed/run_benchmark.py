"""
Main CLI script to orchestrate speed and memory benchmarks across 3Di prediction models
(ESM3Di, ProstT5, ColabFold) and export timing metrics to CSV.

Usage examples:
# Standard dataset size benchmark:
python run_benchmark.py --models colabfold --subsets 10 25 50 100 --no-warmup

# Length-bins benchmark (runtime vs sequence length using binned FASTAs):
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Type

import torch

from models.base import BaseRunner, BenchmarkResult
from models.esm3di_runner import ESM3DiRunner
from models.prostt5_runner import ProstT5Runner
from models.colabfold_runner import ColabFoldRunner


MODEL_REGISTRY: Dict[str, Type[BaseRunner]] = {
    "esm3di": ESM3DiRunner,
    "prostt5": ProstT5Runner,
    "colabfold": ColabFoldRunner,
}


def find_fasta_subsets(data_dir: Path, requested_sizes: List[int] = None) -> List[Path]:
    """Finds and returns sorted subset FASTA files matching subset_<N>.fasta pattern."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Run prepare_data.py first.")

    fasta_files = []
    if requested_sizes:
        for size in requested_sizes:
            target = data_dir / f"subset_{size}.fasta"
            if target.exists():
                fasta_files.append(target)
            else:
                print(f"[!] Warning: Requested subset file not found: {target}")
    else:
        all_subsets = list(data_dir.glob("subset_*.fasta"))

        def extract_size(path: Path) -> int:
            try:
                return int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                return 0

        fasta_files = sorted(all_subsets, key=extract_size)

    if not fasta_files:
        raise FileNotFoundError(f"No FASTA subset files found in '{data_dir}'.")

    return fasta_files


def find_length_bin_fastas(data_dir: Path) -> List[Path]:
    """Finds and returns sorted length-bin FASTA files from data_dir/length_bins/."""
    bin_dir = data_dir / "length_bins"
    if not bin_dir.exists():
        raise FileNotFoundError(
            f"Length bins directory '{bin_dir}' does not exist. "
            "Ensure binned FASTA files (e.g. bin_L100.fasta) are present in that folder."
        )

    bin_files = list(bin_dir.glob("*.fasta"))
    if not bin_files:
        raise FileNotFoundError(f"No FASTA files found in length bins directory '{bin_dir}'.")

    def extract_target_length(path: Path) -> int:
        match = re.search(r"L?(\d+)", path.stem)
        return int(match.group(1)) if match else 0

    return sorted(bin_files, key=extract_target_length)


def append_result_to_csv(
    csv_path: Path,
    result: BenchmarkResult,
    repetition: int,
) -> None:
    """Append one benchmark repetition, extending an existing CSV schema if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = result.to_dict()
    result_dict["repetition"] = repetition

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result_dict.keys()))
            writer.writeheader()
            writer.writerow(result_dict)
        return

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        existing_rows = list(reader)

    fieldnames = list(existing_fieldnames)
    for key in result_dict:
        if key not in fieldnames:
            fieldnames.append(key)

    # Rewrite when the schema changed; this preserves existing results.
    if fieldnames != existing_fieldnames:
        existing_rows.append(result_dict)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
    else:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Run speed and VRAM benchmark across 3Di prediction models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing benchmark subset FASTA files and length_bins subdirectory.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/timing_results.csv"),
        help="Target CSV file for saving metrics.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=["esm3di", "prostt5"],
        help="Models to run.",
    )
    parser.add_argument(
        "--subsets",
        type=int,
        nargs="+",
        default=None,
        help="Specific subset sizes to benchmark (e.g., 10 100 1000). Ignored if --length-bins is set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Target execution device (default: cuda).",
    )
    parser.add_argument(
        "--length-bins",
        action="store_true",
        help="Run length-bins benchmark using multi-sequence FASTA files in data_dir/length_bins/.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of timed inference repetitions per FASTA file.",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be a positive integer.")

    # Determine input dataset mode
    try:
        if args.length_bins:
            #target_files = find_length_bin_fastas(args.data_dir)
            target_files = [Path("data/length_bins/bin_L700.fasta")]
            print(f"[+] Found {len(target_files)} length-bin FASTA file(s) in '{args.data_dir / 'length_bins'}':")
        else:
            target_files = find_fasta_subsets(args.data_dir, args.subsets)
            print(f"[+] Found {len(target_files)} target subset FASTA file(s) in '{args.data_dir}':")

        for f in target_files:
            print(f"    - {f.name}")

    except FileNotFoundError as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Process each selected model
    for model_key in args.models:
        print("\n" + "=" * 60)
        print(f" Running Benchmark Suite for Model: {model_key.upper()}")
        print("=" * 60)

        runner_cls = MODEL_REGISTRY[model_key]
        runner = runner_cls(
            model_name=model_key,
            device=args.device,
        )

        try:
            runner.load_model()
        except Exception as e:
            print(f"[!] Failed to load model '{model_key}': {e}", file=sys.stderr)
            continue

        # Iterate over target FASTA files and repeat each benchmark.
        for fasta_file in target_files:
            for repetition in range(1, args.repeats + 1):
                print(
                    f"\n[+] Executing: {model_key} | File: {fasta_file.name} "
                    f"| Repetition: {repetition}/{args.repeats}"
                )

                try:
                    result = runner.run_benchmark(fasta_file)
                    append_result_to_csv(
                        args.results_file,
                        result,
                        repetition=repetition,
                    )

                    avg_sec_per_seq = (
                        result.wall_time_seconds / result.num_sequences
                        if result.num_sequences > 0
                        else 0.0
                    )

                    print(
                        f"    Done ({result.num_sequences} seqs, {result.total_residues} res) -> "
                        f"Wall Time: {result.wall_time_seconds:.2f}s | "
                        f"Avg/Seq: {avg_sec_per_seq:.3f}s | "
                        f"Res/s: {result.residues_per_second:.1f} | "
                        f"Peak VRAM: {result.peak_vram_gb:.2f} GB"
                    )

                except Exception as e:
                    print(
                        f"[!] Benchmark failed for {model_key} on "
                        f"{fasta_file.name}, repetition {repetition}: {e}",
                        file=sys.stderr,
                    )

        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"[+] Benchmark execution finished. Results saved to: {args.results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
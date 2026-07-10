"""Command Line Interface (CLI) for the ESM3Di Toolkit.

Provides structured subcommands for extracting structural 3Di coordinates,
compiling FoldSeek compatible databases, and exporting per-token model prediction perplexities.
"""

import argparse
import sys
from pathlib import Path

from .io import fasta2foldseek
from .inference import ESM3DiPredictor, DEFAULT_HF_REPO, DEFAULT_BATCH_SIZE


def main():
    """Main execution engine responsible for CLI parsing, basic file verification,

    and high-level routing to underlying library classes.
    """
    parser = argparse.ArgumentParser(
        description="ESM3Di Toolkit: Structural Sequence Predictions and FoldSeek Database Pipelines"
    )

    # 1. Enforce strict subcommand routing
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Target context command route selection"
    )

    # ==========================================
    # Command A: esm3di predict
    # ==========================================
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate structural 3Di string representations from an amino acid FASTA file."
    )
    predict_parser.add_argument(
        "input_fasta",
        help="Path to input amino acid sequence file (.fasta)"
    )
    predict_parser.add_argument(
        "output_fasta",
        help="Destination file path for predicted 3Di labels"
    )
    predict_parser.add_argument(
        "--model-ckpt",
        default=DEFAULT_HF_REPO,
        help=f"Local checkpoint path (.pt) or remote Hugging Face repo ID. (Default: {DEFAULT_HF_REPO})"
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch window size per device (default: {DEFAULT_BATCH_SIZE})"
    )
    predict_parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Explicit hardware card allocation size (default: uses all active cards automatically)"
    )

    # ==========================================
    # Command B: esm3di foldseek
    # ==========================================
    foldseek_parser = subparsers.add_parser(
        "foldseek",
        help="Construct a compiled FoldSeek database directly from sequence inputs."
    )
    foldseek_parser.add_argument(
        "input_fasta",
        help="Path to input amino acid sequence file (.fasta)"
    )
    foldseek_parser.add_argument(
        "output_db",
        help="Target destination directory/prefix pattern path for generating FoldSeek database files"
    )
    foldseek_parser.add_argument(
        "--model-ckpt",
        default=DEFAULT_HF_REPO,
        help=f"Local checkpoint path (.pt) or remote Hugging Face repo ID. (Default: {DEFAULT_HF_REPO})"
    )
    foldseek_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch window size per device (default: {DEFAULT_BATCH_SIZE})"
    )
    foldseek_parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Explicit hardware card allocation size (default: uses all active cards automatically)"
    )

    # ==========================================
    # Command C: esm3di perplexity
    # ==========================================
    perplexity_parser = subparsers.add_parser(
        "perplexity",
        help="Extract structural confidence and per-position token prediction track sequences."
    )
    perplexity_parser.add_argument(
        "input_fasta",
        help="Path to input amino acid sequence file (.fasta)"
    )
    perplexity_parser.add_argument(
        "output_tsv",
        help="Target destination path for compiling tabular data tracking results (.tsv)"
    )
    perplexity_parser.add_argument(
        "--model-ckpt",
        default=DEFAULT_HF_REPO,
        help=f"Local checkpoint path (.pt) or remote Hugging Face repo ID. (Default: {DEFAULT_HF_REPO})"
    )
    perplexity_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch window size (default: {DEFAULT_BATCH_SIZE})"
    )

    args = parser.parse_args()

    # 2. Check resource parameters explicitly utilizing pathlib framework
    input_path = Path(args.input_fasta)
    if not input_path.is_file():
        print(f"✕ Execution Error: Input file could not be verified at track path: '{input_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        # 3. Clean initialization via the newly constructed Factory Method Pattern
        print(f"Initializing ESM3Di engine mapping against checkpoint state: {args.model_ckpt}")
        predictor = ESM3DiPredictor.from_pretrained(args.model_ckpt)

        # Execution Track 1: Standard 3Di evaluation streaming loops
        if args.command == "predict":
            print("Beginning structural 3Di string sequences evaluation pass...")
            predictor.predict_fasta(
                input_fasta_path=args.input_fasta,
                output_fasta_path=args.output_fasta,
                batch_size=args.batch_size,
                num_gpus=args.num_gpus
            )
            print(f"✓ Inference track finished successfully. Result saved to: {args.output_fasta}")

        # Execution Track 2: Compound sequence modeling and database packing
        elif args.command == "foldseek":
            output_db_path = Path(args.output_db)
            # Create transient prediction files co-located with target outputs safely
            temp_3di_fasta = output_db_path.parent / f"{output_db_path.name}_temp_3di.fasta"

            try:
                print("Step 1/2: Resolving multi-track sequence model evaluations...")
                predictor.predict_fasta(
                    input_fasta_path=args.input_fasta,
                    output_fasta_path=temp_3di_fasta,
                    batch_size=args.batch_size,
                    num_gpus=args.num_gpus
                )

                print(f"Step 2/2: Mapping structural representations into FoldSeek layouts at: {args.output_db}")
                fasta2foldseek(
                    aa_input=str(args.input_fasta),
                    tdi_input=str(temp_3di_fasta),
                    output_basename=str(args.output_db)
                )
                print("✓ Binary structural compilation successful. FoldSeek database generated cleanly.")
            finally:
                # Clean up local system resource footprint securely
                if temp_3di_fasta.exists():
                    temp_3di_fasta.unlink()

        # Execution Track 3: Export token confidence data sets
        elif args.command == "perplexity":
            print("Processing per-token track metric algorithms...")
            predictor.output_per_position_perplexity(
                input_fasta_path=args.input_fasta,
                output_tsv_path=args.output_tsv,
                batch_size=args.batch_size
            )
            print(f"✓ Alignment map matrix complete. Records successfully saved to: {args.output_tsv}")

    except Exception as e:
        print(f"✕ Engine Runtime Failure: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
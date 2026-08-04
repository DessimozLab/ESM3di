"""Command Line Interface (CLI) entry point for the ESM3Di Toolkit."""

import argparse
import logging
import sys
import os
import warnings

# Suppress harmless runtime and third-party warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from .io import (
    fasta2foldseek,
    resolve_user_path,
    resolve_output_path,
    resolve_checkpoint_path
)
from .inference import ESM3DiPredictor, DEFAULT_HF_REPO, DEFAULT_BATCH_SIZE, DEFAULT_REVISION

logger = logging.getLogger("esm3di")


def setup_logging():
    """Sets up standard logger format for terminal output."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("esm3di")
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def add_common_args(subparser: argparse.ArgumentParser):
    """Adds arguments shared across subcommands."""
    subparser.add_argument(
        "--model-ckpt",
        default=str(DEFAULT_HF_REPO),
        help=f"Path to local checkpoint or Hugging Face repo ID (default: {DEFAULT_HF_REPO})"
    )
    subparser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Base model Hugging Face revision/commit SHA (default: {DEFAULT_REVISION})"
    )
    subparser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size per device (default: {DEFAULT_BATCH_SIZE})"
    )


def main():
    """Parses command-line arguments and routes commands."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="ESM3Di Toolkit: Predict 3Di structural sequences and build Foldseek databases."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available subcommands"
    )

    # Subcommand: predict
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict 3Di sequences from an amino acid FASTA file and save to FASTA."
    )
    predict_parser.add_argument(
        "--input-fasta",
        default="example_input.fasta",
        help="Path to input amino acid FASTA file (default: example_input.fasta)"
    )
    predict_parser.add_argument(
        "--output-fasta",
        default="outputs/output_3di.fasta",
        help="Path to save output 3Di FASTA file (default: outputs/output_3di.fasta)"
    )
    predict_parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: use all available)"
    )
    add_common_args(predict_parser)

    # Subcommand: build-foldseek-db
    foldseek_parser = subparsers.add_parser(
        "foldseek-db",
        help="Predict 3Di sequences and compile directly into a Foldseek-compatible database."
    )
    foldseek_parser.add_argument(
        "--input-fasta",
        default="example_input.fasta",
        help="Path to input amino acid FASTA file (default: example_input.fasta)"
    )
    foldseek_parser.add_argument(
        "--output-db",
        default="outputs/foldseek_db",
        help="Prefix path for output Foldseek database files (default: outputs/foldseek_db)"
    )
    foldseek_parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: use all available)"
    )
    add_common_args(foldseek_parser)

    # Subcommand: perplexity
    perplexity_parser = subparsers.add_parser(
        "perplexity",
        help="Calculate model confidence (perplexity) for each residue position and export to TSV."
    )
    perplexity_parser.add_argument(
        "--input-fasta",
        default="example_input.fasta",
        help="Path to input amino acid FASTA file (default: example_input.fasta)"
    )
    perplexity_parser.add_argument(
        "--output-tsv",
        default="outputs/output_confidence.tsv",
        help="Path to save output TSV file (default: outputs/output_confidence.tsv)"
    )
    add_common_args(perplexity_parser)

    args = parser.parse_args()

    input_path = resolve_user_path(args.input_fasta)
    model_path_or_id = resolve_checkpoint_path(args.model_ckpt)

    if not input_path.is_file():
        logger.error(f"Input file not found: '{input_path}'")
        sys.exit(1)

    try:
        # Pass the revision down to ESM3DiPredictor
        predictor = ESM3DiPredictor.from_pretrained(
            model_path_or_id,
            revision=args.revision
        )

        if args.command == "predict":
            output_fasta_path = resolve_output_path(args.output_fasta)
            predictor.predict_fasta(
                input_fasta_path=input_path,
                output_fasta_path=output_fasta_path,
                batch_size=args.batch_size,
                num_gpus=args.num_gpus
            )

        elif args.command == "foldseek-db":
            output_db_path = resolve_output_path(args.output_db)
            output_db_path.parent.mkdir(parents=True, exist_ok=True)

            temp_3di_fasta = output_db_path.parent / f"{output_db_path.name}_temp_3di.fasta"

            try:
                predictor.predict_fasta(
                    input_fasta_path=input_path,
                    output_fasta_path=temp_3di_fasta,
                    batch_size=args.batch_size,
                    num_gpus=args.num_gpus
                )

                logger.info(f"Building Foldseek database at: {output_db_path}")
                fasta2foldseek(
                    aa_input=str(input_path),
                    tdi_input=str(temp_3di_fasta),
                    output_basename=str(output_db_path)
                )
                logger.info("Foldseek database generated successfully.")
            finally:
                if temp_3di_fasta.exists():
                    temp_3di_fasta.unlink()

        elif args.command == "perplexity":
            output_tsv_path = resolve_output_path(args.output_tsv)
            predictor.output_per_position_perplexity(
                input_fasta_path=input_path,
                output_tsv_path=output_tsv_path,
                batch_size=args.batch_size
            )

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
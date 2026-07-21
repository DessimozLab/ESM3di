"""Command Line Interface (CLI) for the ESM3Di Toolkit.

Provides structured subcommands for extracting structural 3Di coordinates,
compiling FoldSeek compatible databases, and exporting per-token model prediction perplexities.
"""

import argparse
import logging
import sys
import os
import contextlib
import warnings

# Suppress harmless third-party runtime warnings (e.g., NetworkX duplication)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress default Hugging Face initialization verbosity warnings
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from .io import (
    fasta2foldseek,
    resolve_user_path,
    resolve_output_path,
    resolve_checkpoint_path
)
from .inference import ESM3DiPredictor, DEFAULT_HF_REPO, DEFAULT_BATCH_SIZE

# Configure central logger for CLI execution
logger = logging.getLogger("esm3di")


def setup_logging():
    """Initializes the global logging format for CLI execution."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def main():
    """Main execution engine responsible for CLI parsing and high-level routing."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="ESM3Di Toolkit: Structural Sequence Predictions and FoldSeek Database Pipelines"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Target runtime execution command route track selection"
    )

    # ==========================================
    # Command A: esm3di predict
    # ==========================================
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate structural 3Di string representations from an amino acid FASTA file."
    )
    predict_parser.add_argument(
        "--input-fasta",
        default="example_input.fasta",
        help="Path to input amino acid sequence file (.fasta) (Default: example_input.fasta)"
    )
    predict_parser.add_argument(
        "--output-fasta",
        default="outputs/output_3di.fasta",
        help="Destination file path for output predicted 3Di structural strings (Default: outputs/output_3di.fasta)"
    )
    predict_parser.add_argument(
        "--model-ckpt",
        default=str(DEFAULT_HF_REPO),
        help=f"Local path to hf_compatible folder or remote Hugging Face repo ID. (Default: {DEFAULT_HF_REPO})"
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch tracking window size per hardware device (default: {DEFAULT_BATCH_SIZE})"
    )
    predict_parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Explicit hardware card allocation compute limit size (default: utilizes all active cards automatically)"
    )

    # ==========================================
    # Command B: esm3di foldseek
    # ==========================================
    foldseek_parser = subparsers.add_parser(
        "foldseek",
        help="Construct a compiled FoldSeek database directly from target sequence inputs."
    )
    foldseek_parser.add_argument(
        "--input-fasta",
        default="example_input.fasta",
        help="Path to input amino acid sequence file (.fasta) (Default: example_input.fasta)"
    )
    foldseek_parser.add_argument(
        "--output-db",
        default="outputs/foldseek_db",
        help="Target destination prefix path schema for compilation of structural FoldSeek database objects (Default: outputs/foldseek_db)"
    )
    foldseek_parser.add_argument(
        "--model-ckpt",
        default=str(DEFAULT_HF_REPO),
        help=f"Local path to hf_compatible folder or remote Hugging Face repo ID. (Default: {DEFAULT_HF_REPO})"
    )
    foldseek_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch tracking window size per hardware device (default: {DEFAULT_BATCH_SIZE})"
    )
    foldseek_parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Explicit hardware card allocation compute limit size (default: utilizes all active cards automatically)"
    )

    # ==========================================
    # Command C: esm3di perplexity
    # ==========================================
    perplexity_parser = subparsers.add_parser(
        "perplexity",
        help="Extract token confidence boundaries and per-position tracking prediction perplexities."
    )
    perplexity_parser.add_argument(
        "--input-fasta",
        default="example_input.fasta",
        help="Path to input amino acid sequence file (.fasta) (Default: example_input.fasta)"
    )
    perplexity_parser.add_argument(
        "--output-tsv",
        default="outputs/output_confidence.tsv",
        help="Target destination path for exporting structured tracking logs (.tsv) (Default: outputs/output_confidence.tsv)"
    )
    perplexity_parser.add_argument(
        "--model-ckpt",
        default=str(DEFAULT_HF_REPO),
        help=f"Local path to hf_compatible folder or remote Hugging Face repo ID. (Default: {DEFAULT_HF_REPO})"
    )
    perplexity_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch tracking window size (default: {DEFAULT_BATCH_SIZE})"
    )

    args = parser.parse_args()

    # Resolve input and model files absolutely and robustly
    input_path = resolve_user_path(args.input_fasta)
    model_path_or_id = resolve_checkpoint_path(args.model_ckpt)

    if not input_path.is_file():
        logger.error(f"✕ Verification Failure: Source file could not be found at: '{input_path}'")
        sys.exit(1)

    try:
        logger.info(f"Initializing engine mapping against tracking state target: {model_path_or_id}")
        # Temporarily mute raw print() statements from remote model code (e.g. Synthyra's attention logs)
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            predictor = ESM3DiPredictor.from_pretrained(model_path_or_id)

        # Route Context Track 1: Standard sequence execution mapping
        if args.command == "predict":
            output_fasta_path = resolve_output_path(args.output_fasta)
            output_fasta_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Beginning sequence evaluation loop streaming passes...")
            predictor.predict_fasta(
                input_fasta_path=input_path,
                output_fasta_path=output_fasta_path,
                batch_size=args.batch_size,
                num_gpus=args.num_gpus
            )
            logger.info(f"✓ Inference sequence complete. Records saved to: {output_fasta_path}")

        # Route Context Track 2: Compound sequence alignments and FoldSeek formatting
        elif args.command == "foldseek":
            output_db_path = resolve_output_path(args.output_db)
            output_db_path.parent.mkdir(parents=True, exist_ok=True)

            temp_3di_fasta = output_db_path.parent / f"{output_db_path.name}_temp_3di.fasta"

            try:
                logger.info("Step 1/2: Evaluating sequence tracking pipelines...")
                predictor.predict_fasta(
                    input_fasta_path=input_path,
                    output_fasta_path=temp_3di_fasta,
                    batch_size=args.batch_size,
                    num_gpus=args.num_gpus
                )

                logger.info(f"Step 2/2: Building binary tracking alignment headers at: {output_db_path}")
                fasta2foldseek(
                    aa_input=str(input_path),
                    tdi_input=str(temp_3di_fasta),
                    output_basename=str(output_db_path)
                )
                logger.info("✓ Compilation execution pipeline complete. FoldSeek objects built successfully.")
            finally:
                if temp_3di_fasta.exists():
                    temp_3di_fasta.unlink()

        # Route Context Track 3: Export token confidence metrics
        elif args.command == "perplexity":
            output_tsv_path = resolve_output_path(args.output_tsv)
            output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Processing per-token track algorithms...")
            predictor.output_per_position_perplexity(
                input_fasta_path=input_path,
                output_tsv_path=output_tsv_path,
                batch_size=args.batch_size
            )
            logger.info(f"✓ Evaluation analytics successfully mapped to path target: {output_tsv_path}")

    except Exception as e:
        logger.error(f"✕ Engine Runtime Failure: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
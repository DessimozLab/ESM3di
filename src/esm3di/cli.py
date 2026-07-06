# src/esm3di/cli.py
import argparse
import sys
import os
import torch
from .io import fasta2foldseek
from .inference import ESM3DiPredictor, run_multi_gpu_inference


def main():
    parser = argparse.ArgumentParser(
        description="Generate a FoldSeek database from an amino acid FASTA file using ESM 3Di predictions."
    )
    parser.add_argument(
        "--aa-fasta",
        required=True,
        help="Path to the input amino acid FASTA file."
    )
    parser.add_argument(
        "--output-db",
        required=True,
        help="Target basename for the generated FoldSeek database files."
    )
    # CHANGE THIS: Remove required=True and add default="default"
    parser.add_argument(
        "--model-ckpt",
        default="default",
        help="Path to a local custom model checkpoint (.pt) or Hugging Face repo ID. (Default: Auto-download production weights)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size per GPU (default: 4)."
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to allocate. Defaults to all available GPUs."
    )

    args = parser.parse_args()

    if not os.path.exists(args.aa_fasta):
        print(f"Error: Input FASTA file not found at '{args.aa_fasta}'", file=sys.stderr)
        sys.exit(1)

    num_gpus = torch.cuda.device_count() if args.num_gpus is None else args.num_gpus
    temp_3di_fasta = f"{args.output_db}_temp_3di.fasta"

    try:
        # 1. Inference phase
        if num_gpus > 1:
            print(f"Initializing multi-GPU inference across {num_gpus} targets...")
            # If they are using the default model, pass "default" down to the sharding workers
            success = run_multi_gpu_inference(args.aa_fasta, temp_3di_fasta, args.model_ckpt, num_gpus)
            if not success:
                print("Multi-GPU framework failed or dataset too small. Falling back to single device execution.")
                predictor = ESM3DiPredictor(args.model_ckpt)
                predictor.predict_from_fasta(args.aa_fasta, temp_3di_fasta, batch_size=args.batch_size)
        else:
            print(f"Initializing single-device inference engine on: {'cuda' if torch.cuda.is_available() else 'cpu'}")
            predictor = ESM3DiPredictor(args.model_ckpt)
            predictor.predict_from_fasta(args.aa_fasta, temp_3di_fasta, batch_size=args.batch_size)

        # 2. DB Creation Phase
        print(f"Compiling FoldSeek database files at: {args.output_db}")
        fasta2foldseek(aa_input=args.aa_fasta, tdi_input=temp_3di_fasta, output_basename=args.output_db)
        print("✓ FoldSeek database creation complete.")

    except Exception as e:
        print(f"✕ Critical Error during execution: {str(e)}", file=sys.stderr)
        sys.exit(1)

    finally:
        if os.path.exists(temp_3di_fasta):
            os.remove(temp_3di_fasta)


if __name__ == "__main__":
    main()
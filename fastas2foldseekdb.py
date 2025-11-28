#!/usr/bin/env python
"""
Generate FoldSeek database from amino acid FASTA using ESM 3Di predictions.

This script:
1. Takes an amino acid FASTA file
2. Runs ESM inference to predict 3Di sequences
3. Creates temporary AA and 3Di FASTA files
4. Builds a FoldSeek database using foldseek createdb

Requirements:
- Trained ESM 3Di model checkpoint
- FoldSeek installed and in PATH
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import torch

# Import from our package
from esm3di import predict_3di_for_fasta


def write_fasta(records: List[Tuple[str, str]], output_path: str):
    """Write a list of (header, sequence) tuples to a FASTA file."""
    with open(output_path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def check_foldseek_installed() -> bool:
    """Check if foldseek is available in PATH."""
    try:
        result = subprocess.run(
            ["foldseek", "version"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_foldseek_db(
    aa_fasta: str,
    three_di_fasta: str,
    output_db: str,
    foldseek_bin: str = "foldseek"
) -> bool:
    """
    Create a FoldSeek database from AA and 3Di FASTA files.
    
    Args:
        aa_fasta: Path to amino acid FASTA
        three_di_fasta: Path to 3Di FASTA
        output_db: Path for output database (without extension)
        foldseek_bin: Path to foldseek binary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # FoldSeek createdb expects: foldseek createdb input output
        # For structure databases with 3Di, we use the special format
        cmd = [
            foldseek_bin,
            "createdb",
            aa_fasta,
            output_db,
            "--chain-name-mode", "0",
            "--write-lookup", "1"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        # Now add 3Di information using tsv2db or similar
        # FoldSeek expects 3Di as a separate column/database
        three_di_db = output_db + "_ss"
        cmd_3di = [
            foldseek_bin,
            "createdb",
            three_di_fasta,
            three_di_db,
            "--chain-name-mode", "0",
            "--write-lookup", "1"
        ]
        
        print(f"Running: {' '.join(cmd_3di)}")
        result = subprocess.run(
            cmd_3di,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running foldseek: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate FoldSeek database from AA FASTA using ESM 3Di predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create database from AA FASTA using trained model
  python fastas2foldseekdb.py \\
      --aa-fasta proteins.fasta \\
      --model-ckpt checkpoints/epoch_10.pt \\
      --output-db my_db
  
  # Use pre-computed 3Di FASTA instead of inference
  python fastas2foldseekdb.py \\
      --aa-fasta proteins.fasta \\
      --three-di-fasta proteins_3di.fasta \\
      --output-db my_db \\
      --skip-inference
      
  # Specify custom foldseek binary location
  python fastas2foldseekdb.py \\
      --aa-fasta proteins.fasta \\
      --model-ckpt model.pt \\
      --output-db my_db \\
      --foldseek-bin /path/to/foldseek
"""
    )
    
    # Input files
    parser.add_argument(
        "--aa-fasta",
        type=str,
        required=True,
        help="Input amino acid FASTA file"
    )
    parser.add_argument(
        "--three-di-fasta",
        type=str,
        default=None,
        help="Pre-computed 3Di FASTA file (if --skip-inference)"
    )
    
    # Model and inference
    parser.add_argument(
        "--model-ckpt",
        type=str,
        default=None,
        help="Path to trained ESM 3Di model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference and use provided --three-di-fasta"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda/cpu, auto-detected if not specified)"
    )
    
    # Output
    parser.add_argument(
        "--output-db",
        type=str,
        required=True,
        help="Output FoldSeek database path (without extension)"
    )
    parser.add_argument(
        "--keep-fastas",
        action="store_true",
        help="Keep intermediate AA and 3Di FASTA files"
    )
    parser.add_argument(
        "--output-aa-fasta",
        type=str,
        default=None,
        help="Path to save AA FASTA (default: temp file unless --keep-fastas)"
    )
    parser.add_argument(
        "--output-3di-fasta",
        type=str,
        default=None,
        help="Path to save 3Di FASTA (default: temp file unless --keep-fastas)"
    )
    
    # FoldSeek options
    parser.add_argument(
        "--foldseek-bin",
        type=str,
        default="foldseek",
        help="Path to foldseek binary (default: foldseek in PATH)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.skip_inference and not args.model_ckpt:
        parser.error("--model-ckpt is required unless --skip-inference is set")
    
    if args.skip_inference and not args.three_di_fasta:
        parser.error("--three-di-fasta is required when --skip-inference is set")
    
    if not os.path.exists(args.aa_fasta):
        parser.error(f"Input file not found: {args.aa_fasta}")
    
    if args.skip_inference and not os.path.exists(args.three_di_fasta):
        parser.error(f"3Di FASTA file not found: {args.three_di_fasta}")
    
    if not args.skip_inference and not os.path.exists(args.model_ckpt):
        parser.error(f"Model checkpoint not found: {args.model_ckpt}")
    
    # Check for FoldSeek
    print("Checking for FoldSeek installation...")
    if not check_foldseek_installed():
        print(
            "ERROR: FoldSeek not found in PATH. Please install FoldSeek first.",
            file=sys.stderr
        )
        print("Download from: https://github.com/steineggerlab/foldseek", file=sys.stderr)
        sys.exit(1)
    print("✓ FoldSeek found")
    
    # Set up device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Determine output paths for intermediate FASTAs
    if args.keep_fastas or args.output_aa_fasta or args.output_3di_fasta:
        # Use specified paths or generate defaults
        aa_fasta_out = args.output_aa_fasta or f"{args.output_db}_aa.fasta"
        three_di_fasta_out = args.output_3di_fasta or f"{args.output_db}_3di.fasta"
        use_temp = False
    else:
        # Use temporary files
        use_temp = True
        aa_fasta_out = None
        three_di_fasta_out = None
    
    try:
        # Step 1: Get or generate 3Di predictions
        if args.skip_inference:
            print(f"Using pre-computed 3Di FASTA: {args.three_di_fasta}")
            three_di_fasta_path = args.three_di_fasta
            aa_fasta_path = args.aa_fasta
        else:
            print(f"\nRunning ESM inference on {args.aa_fasta}...")
            print(f"Using model: {args.model_ckpt}")
            print(f"Device: {device}")
            
            # Run inference
            results = predict_3di_for_fasta(
                model_ckpt=args.model_ckpt,
                aa_fasta=args.aa_fasta,
                device=device
            )
            
            print(f"✓ Predicted 3Di for {len(results)} sequences")
            
            # Prepare output
            if use_temp:
                # Create temporary files
                temp_aa = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='_aa.fasta',
                    delete=False
                )
                temp_3di = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='_3di.fasta',
                    delete=False
                )
                aa_fasta_path = temp_aa.name
                three_di_fasta_path = temp_3di.name
                temp_aa.close()
                temp_3di.close()
            else:
                aa_fasta_path = aa_fasta_out
                three_di_fasta_path = three_di_fasta_out
            
            # Write FASTAs
            print(f"\nWriting AA FASTA to: {aa_fasta_path}")
            aa_records = [(header, aa_seq) for header, aa_seq, _ in results]
            write_fasta(aa_records, aa_fasta_path)
            
            print(f"Writing 3Di FASTA to: {three_di_fasta_path}")
            three_di_records = [(header, three_di_seq) for header, _, three_di_seq in results]
            write_fasta(three_di_records, three_di_fasta_path)
            
            print("✓ FASTA files written")
        
        # Step 2: Create FoldSeek database
        print(f"\nCreating FoldSeek database: {args.output_db}")
        success = create_foldseek_db(
            aa_fasta=aa_fasta_path,
            three_di_fasta=three_di_fasta_path,
            output_db=args.output_db,
            foldseek_bin=args.foldseek_bin
        )
        
        if success:
            print(f"\n✓ Successfully created FoldSeek database: {args.output_db}")
            print(f"  - Main database: {args.output_db}")
            print(f"  - 3Di database: {args.output_db}_ss")
            
            # List created files
            db_files = list(Path(args.output_db).parent.glob(f"{Path(args.output_db).name}*"))
            if db_files:
                print(f"\nCreated {len(db_files)} database files:")
                for f in sorted(db_files)[:10]:  # Show first 10
                    print(f"  - {f.name}")
                if len(db_files) > 10:
                    print(f"  ... and {len(db_files) - 10} more")
        else:
            print("\n✗ Failed to create FoldSeek database", file=sys.stderr)
            sys.exit(1)
        
    finally:
        # Clean up temporary files if needed
        if use_temp and not args.skip_inference:
            if aa_fasta_path and os.path.exists(aa_fasta_path):
                os.unlink(aa_fasta_path)
                print(f"Cleaned up temporary file: {aa_fasta_path}")
            if three_di_fasta_path and os.path.exists(three_di_fasta_path):
                os.unlink(three_di_fasta_path)
                print(f"Cleaned up temporary file: {three_di_fasta_path}")
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()

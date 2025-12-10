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
from ESM3di_model import ESM3DiModel


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


def create_foldseek_db_from_fastas(
    aa_fasta: str,
    three_di_fasta: str,
    output_db: str,
    foldseek_bin: str = "foldseek"
) -> bool:
    """
    Create a FoldSeek database from AA and 3Di FASTA files using TSV intermediate format.
    
    Args:
        aa_fasta: Path to amino acid FASTA
        three_di_fasta: Path to 3Di FASTA
        output_db: Path for output database (without extension)
        foldseek_bin: Path to foldseek binary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from Bio import SeqIO
        import tempfile
        
        # Read amino acid sequences
        sequences_aa = {}
        for record in SeqIO.parse(aa_fasta, "fasta"):
            sequences_aa[record.id] = str(record.seq)
        
        # Read 3Di sequences
        sequences_3di = {}
        for record in SeqIO.parse(three_di_fasta, "fasta"):
            if record.id not in sequences_aa.keys():
                print(f"Warning: ignoring 3Di entry {record.id}, since it is not in the amino-acid FASTA file")
            else:
                sequences_3di[record.id] = str(record.seq).upper()
        
        # Validate that all AA sequences have corresponding 3Di sequences
        missing_3di = []
        for seq_id in sequences_aa.keys():
            if seq_id not in sequences_3di.keys():
                missing_3di.append(seq_id)
        
        if missing_3di:
            print(f"Error: {len(missing_3di)} entries in amino-acid FASTA have no corresponding 3Di string:")
            for seq_id in missing_3di[:5]:  # Show first 5
                print(f"  - {seq_id}")
            if len(missing_3di) > 5:
                print(f"  ... and {len(missing_3di) - 5} more")
            return False
        
        # Create temporary TSV files
        with tempfile.NamedTemporaryFile(mode='w', suffix='_aa.tsv', delete=False) as aa_tsv:
            with tempfile.NamedTemporaryFile(mode='w', suffix='_3di.tsv', delete=False) as three_di_tsv:
                with tempfile.NamedTemporaryFile(mode='w', suffix='_header.tsv', delete=False) as header_tsv:
                    
                    # Generate TSV content
                    for i, seq_id in enumerate(sequences_aa.keys(), 1):
                        aa_tsv.write(f"{i}\t{sequences_aa[seq_id]}\n")
                        three_di_tsv.write(f"{i}\t{sequences_3di[seq_id]}\n")
                        header_tsv.write(f"{i}\t{seq_id}\n")
                    
                    aa_tsv_path = aa_tsv.name
                    three_di_tsv_path = three_di_tsv.name
                    header_tsv_path = header_tsv.name
        
        try:
            # Create FoldSeek databases using tsv2db
            commands = [
                [foldseek_bin, "tsv2db", aa_tsv_path, output_db, "--output-dbtype", "0"],
                [foldseek_bin, "tsv2db", three_di_tsv_path, f"{output_db}_ss", "--output-dbtype", "0"],
                [foldseek_bin, "tsv2db", header_tsv_path, f"{output_db}_h", "--output-dbtype", "12"]
            ]
            
            for cmd in commands:
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
            
            print(f"✓ Successfully created FoldSeek database with {len(sequences_aa)} sequences")
            return True
            
        finally:
            # Clean up temporary TSV files
            for temp_file in [aa_tsv_path, three_di_tsv_path, header_tsv_path]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
    except ImportError:
        print("Error: BioPython is required. Install with: pip install biopython", file=sys.stderr)
        return False
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
            
            # Prepare output paths
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
            
            # Load checkpoint to extract model configuration
            print("Loading model configuration from checkpoint...")
            checkpoint = torch.load(args.model_ckpt, map_location="cpu")
            args_dict = checkpoint.get('args', {})
            hf_model_name = args_dict.get(
                'hf_model_name',
                args_dict.get('hf_model', 'facebook/esm2_t33_650M_UR50D')
            )
            num_labels = len(checkpoint.get('label_vocab', []))
            use_cnn_head = args_dict.get('use_cnn_head', False)
            lora_r = args_dict.get('lora_r', 8)
            lora_alpha = args_dict.get('lora_alpha', 16)
            lora_dropout = args_dict.get('lora_dropout', 0.05)
            target_modules = checkpoint.get('lora_target_modules', None)
            
            # Initialize model
            print(f"Initializing ESM3DiModel with {hf_model_name}...")
            model = ESM3DiModel(
                hf_model_name=hf_model_name,
                num_labels=num_labels,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                use_cnn_head=use_cnn_head,
                cnn_num_layers=args_dict.get('cnn_num_layers', 2),
                cnn_kernel_size=args_dict.get('cnn_kernel_size', 3),
                cnn_dropout=args_dict.get('cnn_dropout', 0.1)
            )
            
            # Copy input AA FASTA to output location (for database creation)
            if aa_fasta_path != args.aa_fasta:
                import shutil
                print(f"\nCopying AA FASTA to: {aa_fasta_path}")
                shutil.copy2(args.aa_fasta, aa_fasta_path)
            else:
                print(f"\nUsing input AA FASTA: {aa_fasta_path}")
            
            # Run inference using the model's predict_from_fasta method
            print("Predicting 3Di sequences...")
            model.predict_from_fasta(
                input_fasta_path=args.aa_fasta,
                output_fasta_path=three_di_fasta_path,
                model_checkpoint_path=args.model_ckpt,
                batch_size=4,
                device=device
            )
            
            print("✓ FASTA files ready")
        
        # Step 2: Create FoldSeek database
        print(f"\nCreating FoldSeek database: {args.output_db}")
        success = create_foldseek_db_from_fastas(
            aa_fasta=aa_fasta_path,
            three_di_fasta=three_di_fasta_path,
            output_db=args.output_db,
            foldseek_bin=args.foldseek_bin
        )
        
        if success:
            print(f"\n✓ Successfully created FoldSeek database: "
                  f"{args.output_db}")
            print(f"  - Main database: {args.output_db}")
            print(f"  - 3Di database: {args.output_db}_ss")

            # List created files
            db_name = Path(args.output_db).name
            db_parent = Path(args.output_db).parent
            db_files = list(db_parent.glob(f"{db_name}*"))
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

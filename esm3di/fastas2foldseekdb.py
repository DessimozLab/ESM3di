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
from .ESM3di_model import ESM3DiModel


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

def _count_sequences(fasta_path: str) -> int:
    """
    Count the number of sequences in a FASTA file without loading all into memory.
    
    Uses stream-based parsing for memory efficiency with large files.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Total number of sequences in the file
    """
    from Bio import SeqIO
    count = 0
    for _ in SeqIO.parse(fasta_path, "fasta"):
        count += 1
    return count


def _shard_fasta(input_fasta: str, num_shards: int, temp_dir: str) -> List[Tuple[str, List[str]]]:
    """
    Distribute sequences from a FASTA file across multiple shard files using round-robin.
    
    Sequences are distributed evenly using modulo assignment: sequence i goes to shard (i % num_shards).
    This ensures balanced distribution regardless of file size or sequence order.
    
    Args:
        input_fasta: Path to input amino acid FASTA file
        num_shards: Number of shards to create (typically = number of GPUs)
        temp_dir: Directory for temporary shard files
        
    Returns:
        List of (shard_fasta_path, [header_ids_in_shard]) tuples.
        header_ids_in_shard preserves the order of sequences in this shard.
        This list is needed by _merge_fasta_outputs() to reconstruct original order.
    """
    from Bio import SeqIO
    
    # Initialize shard buffers and header tracking
    shards = [[] for _ in range(num_shards)]  # Each shard stores (header, seq) tuples
    original_order = []  # Track all header IDs in original order
    
    # Read input FASTA and distribute sequences round-robin
    for i, record in enumerate(SeqIO.parse(input_fasta, "fasta")):
        shard_id = i % num_shards
        header = record.id
        sequence = str(record.seq)
        shards[shard_id].append((header, sequence))
        original_order.append(header)
    
    # Write each shard to a temporary FASTA file
    result = []
    for gpu_id in range(num_shards):
        shard_fasta_path = os.path.join(temp_dir, f"shard_{gpu_id}_aa.fasta")
        write_fasta(shards[gpu_id], shard_fasta_path)
        
        # Extract header IDs for this shard (in shard order, NOT original order)
        shard_header_ids = [header for header, _ in shards[gpu_id]]
        result.append((shard_fasta_path, shard_header_ids))
    
    return result


def _merge_fasta_outputs(shard_outputs: List[Tuple[str, str]], output_fasta: str, original_order: List[str]):
    """
    Merge 3Di prediction outputs from multiple shards back into original sequence order.
    
    This function reads 3Di sequences from shard output files and reconstructs the original
    input order. This is critical for database correctness and pipeline reproducibility.
    
    Args:
        shard_outputs: List of (shard_aa_path, shard_3di_path) tuples from worker outputs.
                       Files must exist and contain sequences with matching headers.
        output_fasta: Path to write merged 3Di FASTA in original order
        original_order: List of header IDs in original input order (from _shard_fasta())
        
    Returns:
        None. Writes merged FASTA to output_fasta.
    """
    from Bio import SeqIO
    
    # Read all 3Di predictions into a dictionary keyed by header ID
    all_sequences = {}
    for shard_aa_path, shard_3di_path in shard_outputs:
        # Parse 3Di output file
        for record in SeqIO.parse(shard_3di_path, "fasta"):
            all_sequences[record.id] = str(record.seq)
    
    # Reconstruct output in original order
    entries = [(header, all_sequences[header]) for header in original_order]
    
    # Write merged FASTA
    write_fasta(entries, output_fasta)


def _gpu_worker(gpu_id, shard_fasta, output_fasta, checkpoint_path, args_dict, progress_queue, error_event):
    """
    GPU worker function for multi-GPU inference with CUDA isolation.
    
    This function runs in a spawned subprocess. It isolates GPU access by setting
    CUDA_VISIBLE_DEVICES to a single GPU before importing torch, ensuring safe
    multi-GPU inference without CUDA context conflicts.
    
    Args:
        gpu_id: GPU index to use (0-based)
        shard_fasta: Path to input amino acid FASTA for this worker
        output_fasta: Path to write predicted 3Di FASTA
        checkpoint_path: Path to model checkpoint
        args_dict: Dictionary with model args (hf_model, lora_r, lora_alpha, etc.)
        progress_queue: multiprocessing.Queue for sending progress/completion events
        error_event: multiprocessing.Event to signal errors
    
    Sends to progress_queue:
        - ("progress", gpu_id, sequences_processed, total_sequences)
        - ("done", gpu_id)
        - ("error", gpu_id, error_message) [also sets error_event]
    """
    try:
        # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Delayed import: torch must be imported AFTER CUDA_VISIBLE_DEVICES is set
        import torch
        
        # Set CPU thread limiting to prevent 30x performance degradation
        # Calculate threads per GPU based on total CPU count
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        num_gpus = max(1, num_gpus)  # At least 1
        cpu_count = os.cpu_count() or 1
        threads_per_gpu = max(1, cpu_count // num_gpus)
        torch.set_num_threads(threads_per_gpu)
        
        # Set device to cuda (will be gpu 0 due to CUDA_VISIBLE_DEVICES)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load checkpoint to extract model configuration
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract model initialization parameters
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
        
        # Run inference on this shard
        model.predict_from_fasta(
            input_fasta_path=shard_fasta,
            output_fasta_path=output_fasta,
            model_checkpoint_path=checkpoint_path,
            batch_size=4,
            device=device
        )
        
        # Signal completion
        progress_queue.put(("done", gpu_id))
        
    except Exception as e:
        # On error: send error message and signal error event
        error_message = f"GPU {gpu_id} worker failed: {str(e)}"
        progress_queue.put(("error", gpu_id, error_message))
        error_event.set()


def _run_multi_gpu_inference(input_fasta, output_3di_fasta, checkpoint_path, args_dict, num_gpus):
    """
    Coordinate multi-GPU inference by sharding input and running parallel workers.
    
    This function distributes sequences across multiple GPUs using process-per-GPU
    architecture. Each worker runs in isolation with CUDA_VISIBLE_DEVICES set to
    a single GPU.
    
    Args:
        input_fasta: Path to input amino acid FASTA file
        output_3di_fasta: Path to write merged 3Di predictions
        checkpoint_path: Path to model checkpoint
        args_dict: Dictionary with model args (hf_model, lora_r, lora_alpha, etc.)
        num_gpus: Number of GPUs to use for inference
        
    Returns:
        True if multi-GPU inference completed successfully.
        False if single-GPU fallback should be used (num_sequences==1 or num_gpus==1).
        
    Raises:
        ValueError: If input FASTA is empty
        RuntimeError: If any worker fails
    """
    import multiprocessing as mp
    import queue
    import shutil
    from tqdm import tqdm
    from Bio import SeqIO
    
    # Use spawn context for CUDA compatibility
    # (fork doesn't work with CUDA - "Cannot re-initialize CUDA in forked subprocess")
    ctx = mp.get_context('spawn')
    
    # 1. Count sequences - early exit conditions
    num_sequences = _count_sequences(input_fasta)
    if num_sequences == 0:
        raise ValueError("Input FASTA is empty")
    if num_sequences == 1 or num_gpus == 1:
        return False  # Signal: use single-GPU path
    
    # 2. Get original sequence order for merge (must preserve input order)
    original_order = [rec.id for rec in SeqIO.parse(input_fasta, "fasta")]
    
    # 3. Create temp directory for shards
    temp_dir = tempfile.mkdtemp(prefix="esm3di_shards_")
    
    try:
        # 4. Shard input FASTA across GPUs
        shards = _shard_fasta(input_fasta, num_gpus, temp_dir)
        
        # 5. Create IPC primitives (using spawn context)
        progress_queue = ctx.Queue()
        error_event = ctx.Event()
        
        # 6. Prepare worker outputs and spawn processes
        shard_outputs = []
        processes = []
        for gpu_id, (shard_fasta, shard_headers) in enumerate(shards):
            output_path = os.path.join(temp_dir, f"shard_{gpu_id}_3di.fasta")
            shard_outputs.append((shard_fasta, output_path))
            
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, shard_fasta, output_path, checkpoint_path, args_dict, progress_queue, error_event)
            )
            processes.append(p)
            p.start()
        
        print(f"Using {num_gpus} GPU(s) for inference on {num_sequences} sequences")
        
        # 7. Progress monitoring loop
        completed = 0
        with tqdm(total=num_gpus, desc="Multi-GPU inference", unit="GPU") as pbar:
            while completed < num_gpus:
                try:
                    event = progress_queue.get(timeout=0.5)
                    if event[0] == "done":
                        completed += 1
                        pbar.update(1)
                    elif event[0] == "error":
                        # Terminate all workers on error
                        for p in processes:
                            p.terminate()
                        raise RuntimeError(event[2])
                except queue.Empty:
                    pass
                
                # Check error event after each poll
                if error_event.is_set():
                    for p in processes:
                        p.terminate()
                    # Drain queue for error message
                    try:
                        while True:
                            event = progress_queue.get_nowait()
                            if event[0] == "error":
                                raise RuntimeError(event[2])
                    except queue.Empty:
                        raise RuntimeError("Worker signaled error")
        
        # 8. Wait for all processes to finish
        for p in processes:
            p.join()
        
        # 9. Merge outputs in original order
        _merge_fasta_outputs(shard_outputs, output_3di_fasta, original_order)
        
        return True
        
    finally:
        # 10. Cleanup temp shard files
        shutil.rmtree(temp_dir, ignore_errors=True)

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
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for inference (default: auto-detect all available)"
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
    parser.add_argument(
        "--output-confidence-fasta",
        type=str,
        default=None,
        help="Path to save confidence FASTA (optional, default: not saved)"
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
            
            # AA fasta path is always the input file
            aa_fasta_path = args.aa_fasta
            
            # Prepare output path for 3Di
            if use_temp:
                # Create temporary file for 3Di output
                temp_3di = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='_3di.fasta',
                    delete=False
                )
                three_di_fasta_path = temp_3di.name
                temp_3di.close()
            else:
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
            label_vocab = checkpoint.get('label_vocab', [])
            idx2char = {i: c for i, c in enumerate(label_vocab)}
            
            use_cnn_head = args_dict.get('use_cnn_head', False)
            lora_r = args_dict.get('lora_r', 8)
            lora_alpha = args_dict.get('lora_alpha', 16)
            lora_dropout = args_dict.get('lora_dropout', 0.05)
            target_modules = checkpoint.get('lora_target_modules', None)
            
            # Determine number of GPUs to use
            num_gpus = torch.cuda.device_count() if args.num_gpus is None else args.num_gpus
            
            # Warn if --device flag is ignored in multi-GPU mode
            if num_gpus > 1 and args.device is not None:
                print(f"WARNING: --device flag ignored in multi-GPU mode (using {num_gpus} GPUs)")
            
            # Build args_dict for workers
            coordinator_args_dict = {
                'hf_model': hf_model_name,
                'hf_model_name': hf_model_name,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout,
                'use_cnn_head': use_cnn_head,
                'cnn_num_layers': args_dict.get('cnn_num_layers', 2),
                'cnn_kernel_size': args_dict.get('cnn_kernel_size', 3),
                'cnn_dropout': args_dict.get('cnn_dropout', 0.1),
                'batch_size': 4
            }
            
            # Try multi-GPU inference first
            use_multi_gpu = False
            if num_gpus > 1:
                try:
                    use_multi_gpu = _run_multi_gpu_inference(
                        input_fasta=args.aa_fasta,
                        output_3di_fasta=three_di_fasta_path,
                        checkpoint_path=args.model_ckpt,
                        args_dict=coordinator_args_dict,
                        num_gpus=num_gpus
                    )
                except Exception as e:
                    print(f"WARNING: Multi-GPU inference failed: {e}")
                    print("Falling back to single-GPU inference...")
                    use_multi_gpu = False
            
            # Single-GPU fallback
            if not use_multi_gpu:
                print(f"Using single GPU: {device}")
                
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
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()

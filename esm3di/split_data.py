#!/usr/bin/env python
"""
Utility script to split paired AA/3Di FASTA files into train/test/validation sets.
Optionally also splits a pLDDT bins FASTA file.

Usage:
    python -m esm3di.split_data \
        --aa-fasta data_aa.fasta \
        --three-di-fasta data_3di.fasta \
        --plddt-bins-fasta data_plddt_bins.fasta \
        --output-prefix split_data \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --test-ratio 0.1
"""

import argparse
import random
import os
from typing import List, Tuple, Dict


def read_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """Read FASTA file and return list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))
    
    return sequences


def write_fasta(sequences: List[Tuple[str, str]], output_path: str, line_width: int = 80):
    """Write sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for header, seq in sequences:
            f.write(f">{header}\n")
            # Wrap sequence to line_width characters
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + '\n')


def validate_paired_fastas(aa_seqs: List[Tuple[str, str]], 
                           threedi_seqs: List[Tuple[str, str]],
                           plddt_seqs: List[Tuple[str, str]] = None) -> None:
    """Validate that AA, 3Di, and optional pLDDT FASTAs have matching headers and lengths."""
    if len(aa_seqs) != len(threedi_seqs):
        raise ValueError(
            f"Number of sequences mismatch: AA has {len(aa_seqs)}, "
            f"3Di has {len(threedi_seqs)}"
        )
    
    if plddt_seqs is not None and len(plddt_seqs) != len(aa_seqs):
        raise ValueError(
            f"Number of sequences mismatch: AA has {len(aa_seqs)}, "
            f"pLDDT has {len(plddt_seqs)}"
        )
    
    for i, ((aa_header, aa_seq), (di_header, di_seq)) in enumerate(zip(aa_seqs, threedi_seqs)):
        if aa_header != di_header:
            raise ValueError(
                f"Header mismatch at position {i}: '{aa_header}' vs '{di_header}'"
            )
        if len(aa_seq) != len(di_seq):
            raise ValueError(
                f"Sequence length mismatch for '{aa_header}': "
                f"AA={len(aa_seq)}, 3Di={len(di_seq)}"
            )
        
        if plddt_seqs is not None:
            plddt_header, plddt_seq = plddt_seqs[i]
            if aa_header != plddt_header:
                raise ValueError(
                    f"pLDDT header mismatch at position {i}: '{aa_header}' vs '{plddt_header}'"
                )
            if len(aa_seq) != len(plddt_seq):
                raise ValueError(
                    f"pLDDT length mismatch for '{aa_header}': "
                    f"AA={len(aa_seq)}, pLDDT={len(plddt_seq)}"
                )


def split_data(
    aa_seqs: List[Tuple[str, str]],
    threedi_seqs: List[Tuple[str, str]],
    plddt_seqs: List[Tuple[str, str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Tuple[List, ...]]:
    """
    Split paired sequences into train/val/test sets.
    
    Returns:
        Dict with keys 'train', 'val', 'test', each containing
        (aa_sequences, threedi_sequences) or (aa_sequences, threedi_sequences, plddt_sequences) tuple
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    n = len(aa_seqs)
    
    # Create indices and shuffle
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    
    # Calculate split points
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create splits
    splits = {}
    for name, idx_list in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        aa_split = [aa_seqs[i] for i in idx_list]
        di_split = [threedi_seqs[i] for i in idx_list]
        if plddt_seqs is not None:
            plddt_split = [plddt_seqs[i] for i in idx_list]
            splits[name] = (aa_split, di_split, plddt_split)
        else:
            splits[name] = (aa_split, di_split)
    
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Split paired AA/3Di FASTA files into train/test/validation sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--aa-fasta", type=str, required=True,
                        help="Input FASTA with amino-acid sequences")
    parser.add_argument("--three-di-fasta", type=str, required=True,
                        help="Input FASTA with matching 3Di sequences")
    parser.add_argument("--plddt-bins-fasta", type=str, default=None,
                        help="Optional input FASTA with pLDDT bins (0-9 per position)")
    
    # Output
    parser.add_argument("--output-prefix", type=str, required=True,
                        help="Prefix for output files. Will create "
                             "{prefix}_train_aa.fasta, {prefix}_train_3di.fasta, etc.")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for split files")
    
    # Split ratios
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of data for training")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of data for testing")
    
    # Options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip test set (only create train and val)")
    
    args = parser.parse_args()
    
    # Adjust ratios if no test set
    if args.no_test:
        args.test_ratio = 0.0
        # Renormalize train and val ratios
        total = args.train_ratio + args.val_ratio
        args.train_ratio = args.train_ratio / total
        args.val_ratio = args.val_ratio / total
        print(f"No test set requested. Adjusted ratios: "
              f"train={args.train_ratio:.2f}, val={args.val_ratio:.2f}")
    
    # Validate ratios sum to 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {total_ratio:.4f}")
    
    print(f"\n{'='*60}")
    print("FASTA Train/Val/Test Splitter")
    print(f"{'='*60}")
    
    # Read input files
    print(f"\nReading AA FASTA: {args.aa_fasta}")
    aa_seqs = read_fasta(args.aa_fasta)
    print(f"  Found {len(aa_seqs)} sequences")
    
    print(f"Reading 3Di FASTA: {args.three_di_fasta}")
    threedi_seqs = read_fasta(args.three_di_fasta)
    print(f"  Found {len(threedi_seqs)} sequences")
    
    # Read optional pLDDT bins
    plddt_seqs = None
    if args.plddt_bins_fasta:
        print(f"Reading pLDDT bins FASTA: {args.plddt_bins_fasta}")
        plddt_seqs = read_fasta(args.plddt_bins_fasta)
        print(f"  Found {len(plddt_seqs)} sequences")
    
    # Validate
    print("\nValidating paired sequences...")
    validate_paired_fastas(aa_seqs, threedi_seqs, plddt_seqs)
    print("  ✓ All sequences match")
    
    # Split
    print(f"\nSplitting with ratios: train={args.train_ratio:.1%}, "
          f"val={args.val_ratio:.1%}, test={args.test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    
    splits = split_data(
        aa_seqs, threedi_seqs,
        plddt_seqs=plddt_seqs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write output files
    print(f"\nWriting output files to: {args.output_dir}")
    
    has_plddt = plddt_seqs is not None
    
    for split_name, split_data_tuple in splits.items():
        if has_plddt:
            aa_split, di_split, plddt_split = split_data_tuple
        else:
            aa_split, di_split = split_data_tuple
            plddt_split = None
        
        if len(aa_split) == 0:
            print(f"  {split_name}: skipped (0 sequences)")
            continue
            
        aa_path = os.path.join(args.output_dir, f"{args.output_prefix}_{split_name}_aa.fasta")
        di_path = os.path.join(args.output_dir, f"{args.output_prefix}_{split_name}_3di.fasta")
        
        write_fasta(aa_split, aa_path)
        write_fasta(di_split, di_path)
        
        print(f"  {split_name}: {len(aa_split)} sequences")
        print(f"    → {aa_path}")
        print(f"    → {di_path}")
        
        if plddt_split is not None:
            plddt_path = os.path.join(args.output_dir, f"{args.output_prefix}_{split_name}_plddt_bins.fasta")
            write_fasta(plddt_split, plddt_path)
            print(f"    → {plddt_path}")
    
    print(f"\n{'='*60}")
    print("Split complete!")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    total = len(aa_seqs)
    for split_name, split_tuple in splits.items():
        n = len(split_tuple[0])  # aa_split is always first
        pct = 100 * n / total if total > 0 else 0
        print(f"  {split_name:5s}: {n:6d} sequences ({pct:5.1f}%)")
    
    print(f"\nTo use with training:")
    train_aa = os.path.join(args.output_dir, f"{args.output_prefix}_train_aa.fasta")
    train_3di = os.path.join(args.output_dir, f"{args.output_prefix}_train_3di.fasta")
    val_aa = os.path.join(args.output_dir, f"{args.output_prefix}_val_aa.fasta")
    val_3di = os.path.join(args.output_dir, f"{args.output_prefix}_val_3di.fasta")
    
    print(f"  python -m esm3di.esmretrain \\")
    print(f"    --aa-fasta {train_aa} \\")
    print(f"    --three-di-fasta {train_3di} \\")
    if has_plddt:
        train_plddt = os.path.join(args.output_dir, f"{args.output_prefix}_train_plddt_bins.fasta")
        val_plddt = os.path.join(args.output_dir, f"{args.output_prefix}_val_plddt_bins.fasta")
        print(f"    --plddt-bins-fasta {train_plddt} \\")
        print(f"    --val-aa-fasta {val_aa} \\")
        print(f"    --val-three-di-fasta {val_3di} \\")
        print(f"    --val-plddt-bins-fasta {val_plddt}")
    else:
        print(f"    --val-aa-fasta {val_aa} \\")
        print(f"    --val-three-di-fasta {val_3di}")


if __name__ == "__main__":
    main()

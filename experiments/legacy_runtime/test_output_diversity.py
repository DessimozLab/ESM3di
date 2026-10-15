#!/usr/bin/env python
"""
Test script to verify 3Di predictions have sufficient diversity.

This script checks that:
1. Output FASTA files are not all the same character
2. Character distribution is reasonable for 3Di alphabet
3. Compares outputs from different checkpoints
"""

import argparse
import sys
from collections import Counter
from pathlib import Path


def read_fasta_sequences(fasta_path: str) -> dict:
    """Read sequences from a FASTA file."""
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_header is not None:
            sequences[current_header] = ''.join(current_seq)
    
    return sequences


def analyze_diversity(sequences: dict) -> dict:
    """Analyze character diversity in sequences."""
    all_chars = ''.join(sequences.values())
    total_len = len(all_chars)
    
    if total_len == 0:
        return {
            'total_chars': 0,
            'unique_chars': 0,
            'distribution': {},
            'is_uniform': True,
            'dominant_char': None,
            'dominant_pct': 0.0
        }
    
    char_counts = Counter(all_chars)
    unique_chars = len(char_counts)
    
    # Find dominant character
    most_common = char_counts.most_common(1)[0]
    dominant_char = most_common[0]
    dominant_pct = most_common[1] / total_len * 100
    
    # Calculate distribution
    distribution = {
        char: count / total_len * 100 
        for char, count in char_counts.most_common()
    }
    
    # Check if output is effectively uniform (>90% one character)
    is_uniform = dominant_pct > 90.0
    
    return {
        'total_chars': total_len,
        'unique_chars': unique_chars,
        'distribution': distribution,
        'is_uniform': is_uniform,
        'dominant_char': dominant_char,
        'dominant_pct': dominant_pct
    }


def check_diversity(fasta_path: str, min_unique_chars: int = 5, 
                   max_dominant_pct: float = 50.0) -> bool:
    """
    Check if a 3Di FASTA file has sufficient diversity.
    
    Args:
        fasta_path: Path to 3Di FASTA file
        min_unique_chars: Minimum number of unique characters expected
        max_dominant_pct: Maximum allowed percentage for a single character
        
    Returns:
        True if diversity is acceptable, False otherwise
    """
    print(f"\nAnalyzing: {fasta_path}")
    
    if not Path(fasta_path).exists():
        print(f"  ERROR: File not found")
        return False
    
    sequences = read_fasta_sequences(fasta_path)
    
    if len(sequences) == 0:
        print(f"  ERROR: No sequences found in file")
        return False
    
    print(f"  Found {len(sequences)} sequences")
    
    analysis = analyze_diversity(sequences)
    
    print(f"  Total residues: {analysis['total_chars']}")
    print(f"  Unique characters: {analysis['unique_chars']}")
    print(f"  Dominant character: '{analysis['dominant_char']}' ({analysis['dominant_pct']:.1f}%)")
    
    print(f"\n  Character distribution (top 10):")
    for i, (char, pct) in enumerate(analysis['distribution'].items()):
        if i >= 10:
            break
        bar_len = int(pct / 2)
        bar = '█' * bar_len
        print(f"    {char}: {pct:5.1f}% {bar}")
    
    # Check criteria
    passed = True
    
    if analysis['unique_chars'] < min_unique_chars:
        print(f"\n  FAIL: Only {analysis['unique_chars']} unique chars (need >= {min_unique_chars})")
        passed = False
    
    if analysis['dominant_pct'] > max_dominant_pct:
        print(f"\n  FAIL: Dominant char is {analysis['dominant_pct']:.1f}% (max allowed: {max_dominant_pct}%)")
        passed = False
    
    if analysis['is_uniform']:
        print(f"\n  FAIL: Output is effectively uniform (>90% one character)")
        passed = False
    
    if passed:
        print(f"\n  PASS: Output has acceptable diversity")
    
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Test 3Di prediction output diversity"
    )
    parser.add_argument(
        "fasta_files",
        nargs="+",
        help="3Di FASTA file(s) to analyze"
    )
    parser.add_argument(
        "--min-unique",
        type=int,
        default=5,
        help="Minimum number of unique characters (default: 5)"
    )
    parser.add_argument(
        "--max-dominant",
        type=float,
        default=50.0,
        help="Maximum percentage for dominant character (default: 50%%)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("3Di Output Diversity Test")
    print("=" * 60)
    
    all_passed = True
    
    for fasta_path in args.fasta_files:
        passed = check_diversity(
            fasta_path,
            min_unique_chars=args.min_unique,
            max_dominant_pct=args.max_dominant
        )
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: All files passed diversity check")
        sys.exit(0)
    else:
        print("OVERALL: Some files failed diversity check")
        sys.exit(1)


if __name__ == "__main__":
    main()

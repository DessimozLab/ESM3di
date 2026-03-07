#!/usr/bin/env python3
"""
Create a taxonomically balanced subset of BFVD data.

Balances the dataset by sampling equal numbers of sequences from each taxon
at a specified taxonomic level (phylum, class, family, etc.).

Example usage:
    # Balance by phylum, sampling up to 500 sequences per phylum
    python -m esm3di.balance_by_taxon --level phylum --max-per-taxon 500 \
        --output-prefix /mnt/data1/bfvd/balanced/phylum_balanced

    # Balance by class with minimum 50 sequences per class
    python -m esm3di.balance_by_taxon --level class --max-per-taxon 1000 \
        --min-per-taxon 50 --output-prefix balanced_class

    # Use the minimum taxon count as the sample size (perfect balance)
    python -m esm3di.balance_by_taxon --level phylum --use-min \
        --output-prefix perfectly_balanced
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


# Default paths
DEFAULT_METADATA = "/mnt/data1/bfvd/metadata/bfvd_taxid_rank_scientificname_lineage.tsv"
DEFAULT_AA_FASTA = "/mnt/data1/bfvd/training_set/bfvd_data_aa.fasta"
DEFAULT_3DI_FASTA = "/mnt/data1/bfvd/training_set/bfvd_data_3di.fasta"
DEFAULT_3DI_MASKED_FASTA = "/mnt/data1/bfvd/training_set/bfvd_data_3di_masked.fasta"

LEVEL_PREFIXES = {
    'domain': 'd_',
    'kingdom': 'k_',
    'phylum': 'p_',
    'class': 'c_',
    'order': 'o_',
    'family': 'f_',
    'genus': 'g_',
    'species': 's_'
}


def parse_lineage_level(lineage: str, level: str) -> Optional[str]:
    """Extract taxon name at a given level from lineage string."""
    prefix = LEVEL_PREFIXES.get(level)
    if not prefix:
        return None
    
    for part in lineage.split(';'):
        if part.startswith(prefix):
            return part[2:]  # Remove prefix
    return None


def group_accessions_by_taxon(
    metadata_path: str,
    level: str,
    verbose: bool = False
) -> Dict[str, List[str]]:
    """Group accessions by taxon at the specified level.
    
    Returns:
        Dict mapping taxon name to list of accession IDs
    """
    taxon_accessions = defaultdict(list)
    
    with open(metadata_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            
            pdb_filename = parts[0]
            lineage = parts[4] if len(parts) > 4 else ''
            
            # Extract accession
            accession = pdb_filename.split('_')[0] if '_' in pdb_filename else pdb_filename.replace('.pdb', '')
            
            # Get taxon at level
            taxon = parse_lineage_level(lineage, level)
            if taxon:
                taxon_accessions[taxon].append(accession)
            
            if verbose and line_num % 100000 == 0:
                print(f"Processed {line_num:,} entries...", file=sys.stderr)
    
    return dict(taxon_accessions)


def sample_balanced(
    taxon_accessions: Dict[str, List[str]],
    max_per_taxon: Optional[int] = None,
    min_per_taxon: int = 0,
    use_min: bool = False,
    seed: int = 42
) -> Tuple[Set[str], Dict[str, int]]:
    """Sample balanced set of accessions.
    
    Args:
        taxon_accessions: Dict mapping taxon to list of accessions
        max_per_taxon: Maximum sequences to sample per taxon
        min_per_taxon: Minimum sequences required (taxa with fewer are excluded)
        use_min: If True, sample exactly min(counts) from each taxon
        seed: Random seed
        
    Returns:
        Tuple of (set of selected accessions, dict of taxon: count sampled)
    """
    random.seed(seed)
    
    # Filter taxa by minimum count
    filtered_taxa = {
        taxon: accs for taxon, accs in taxon_accessions.items()
        if len(accs) >= min_per_taxon
    }
    
    if not filtered_taxa:
        return set(), {}
    
    # Determine sample size
    if use_min:
        sample_size = min(len(accs) for accs in filtered_taxa.values())
    elif max_per_taxon:
        sample_size = max_per_taxon
    else:
        sample_size = min(len(accs) for accs in filtered_taxa.values())
    
    selected = set()
    sampled_counts = {}
    
    for taxon, accessions in sorted(filtered_taxa.items()):
        # Deduplicate accessions (same protein may appear multiple times)
        unique_accs = list(set(accessions))
        
        # Sample
        n_sample = min(sample_size, len(unique_accs))
        sampled = random.sample(unique_accs, n_sample)
        
        selected.update(sampled)
        sampled_counts[taxon] = n_sample
    
    return selected, sampled_counts


def extract_sequences_streaming(
    input_fasta: str,
    output_fasta: str,
    filter_accessions: Set[str],
    verbose: bool = False
) -> int:
    """Extract sequences matching filter set in streaming fashion."""
    count = 0
    include_current = False
    
    with open(input_fasta, 'r') as fin, open(output_fasta, 'w') as fout:
        for line in fin:
            if line.startswith('>'):
                header = line[1:].strip()
                accession = header.split()[0].split('_')[0]
                include_current = accession in filter_accessions
                
                if include_current:
                    fout.write(line)
                    count += 1
            elif include_current:
                fout.write(line)
    
    if verbose:
        print(f"Wrote {count:,} sequences to {output_fasta}", file=sys.stderr)
    
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Create a taxonomically balanced BFVD subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--output-prefix', '-o', type=str, required=True,
                        help='Output prefix for FASTA files')
    
    # Balancing options
    parser.add_argument('--level', '-l', type=str, default='phylum',
                        choices=list(LEVEL_PREFIXES.keys()),
                        help='Taxonomic level to balance on (default: phylum)')
    parser.add_argument('--max-per-taxon', '-n', type=int, default=None,
                        help='Maximum sequences per taxon (default: use minimum count)')
    parser.add_argument('--min-per-taxon', type=int, default=0,
                        help='Exclude taxa with fewer than this many sequences')
    parser.add_argument('--use-min', action='store_true',
                        help='Sample exactly the minimum taxon count from each (perfect balance)')
    
    # Path options
    parser.add_argument('--metadata', '-m', type=str, default=DEFAULT_METADATA,
                        help=f'Path to metadata TSV file')
    parser.add_argument('--aa-fasta', type=str, default=DEFAULT_AA_FASTA,
                        help='Path to amino acid FASTA')
    parser.add_argument('--three-di-fasta', type=str, default=DEFAULT_3DI_FASTA,
                        help='Path to 3Di FASTA')
    parser.add_argument('--use-masked', action='store_true',
                        help='Use masked 3Di FASTA instead')
    parser.add_argument('--masked-fasta', type=str, default=DEFAULT_3DI_MASKED_FASTA,
                        help='Path to masked 3Di FASTA')
    
    # General options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress information')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be sampled without extracting')
    parser.add_argument('--split', action='store_true',
                        help='Also create train/val/test splits')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Use masked if requested
    three_di_fasta = args.masked_fasta if args.use_masked else args.three_di_fasta
    
    print(f"Grouping sequences by {args.level}...", file=sys.stderr)
    taxon_accessions = group_accessions_by_taxon(
        args.metadata, args.level, verbose=args.verbose
    )
    
    print(f"\nFound {len(taxon_accessions)} {args.level}-level taxa:", file=sys.stderr)
    
    # Show distribution
    sorted_taxa = sorted(taxon_accessions.items(), key=lambda x: -len(x[1]))
    for taxon, accs in sorted_taxa[:15]:
        unique_count = len(set(accs))
        print(f"  {taxon}: {unique_count:,} unique accessions", file=sys.stderr)
    if len(sorted_taxa) > 15:
        print(f"  ... and {len(sorted_taxa) - 15} more taxa", file=sys.stderr)
    
    # Sample balanced set
    print(f"\nSampling balanced set...", file=sys.stderr)
    selected_accessions, sampled_counts = sample_balanced(
        taxon_accessions,
        max_per_taxon=args.max_per_taxon,
        min_per_taxon=args.min_per_taxon,
        use_min=args.use_min,
        seed=args.seed
    )
    
    print(f"\nBalanced sampling results:", file=sys.stderr)
    for taxon, count in sorted(sampled_counts.items()):
        print(f"  {taxon}: {count:,} sequences", file=sys.stderr)
    
    total = sum(sampled_counts.values())
    print(f"\nTotal: {total:,} sequences from {len(sampled_counts)} taxa", file=sys.stderr)
    
    if args.dry_run:
        print("\nDry run - no files written", file=sys.stderr)
        return
    
    # Create output directory
    output_dir = Path(args.output_prefix).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract sequences
    aa_output = f"{args.output_prefix}_aa.fasta"
    threedi_output = f"{args.output_prefix}_3di.fasta"
    
    print(f"\nExtracting sequences...", file=sys.stderr)
    extract_sequences_streaming(args.aa_fasta, aa_output, selected_accessions, verbose=args.verbose)
    extract_sequences_streaming(three_di_fasta, threedi_output, selected_accessions, verbose=args.verbose)
    
    print(f"\nOutput files:", file=sys.stderr)
    print(f"  {aa_output}", file=sys.stderr)
    print(f"  {threedi_output}", file=sys.stderr)
    
    # Create splits if requested
    if args.split:
        print(f"\nCreating train/val/test splits...", file=sys.stderr)
        import subprocess
        split_cmd = [
            sys.executable, '-m', 'esm3di.split_data',
            '--aa-fasta', aa_output,
            '--three-di-fasta', threedi_output,
            '--output-prefix', Path(args.output_prefix).name,
            '--output-dir', str(output_dir) if output_dir else '.',
            '--train-ratio', str(args.train_ratio),
            '--val-ratio', str(args.val_ratio),
            '--seed', str(args.seed)
        ]
        subprocess.run(split_cmd, check=True)
    
    print("\nDone!", file=sys.stderr)


if __name__ == '__main__':
    main()

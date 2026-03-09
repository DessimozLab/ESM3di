#!/usr/bin/env python3
"""
Extract a subset of BFVD data based on taxonomic criteria.

This script filters the BFVD metadata by taxon name (searching in scientific name
or lineage) and extracts the corresponding sequences from the AA and 3Di FASTA files.

Example usage:
    # Extract all Coronaviridae
    python -m esm3di.extract_taxon_subset --taxon "Coronaviridae" --output-prefix coronaviridae

    # Extract by genus
    python -m esm3di.extract_taxon_subset --taxon "Betacoronavirus" --output-prefix betacov

    # Use custom paths
    python -m esm3di.extract_taxon_subset --taxon "Flaviviridae" \
        --metadata /path/to/metadata.tsv \
        --aa-fasta /path/to/aa.fasta \
        --three-di-fasta /path/to/3di.fasta \
        --output-prefix flavi

    # Case-sensitive exact match
    python -m esm3di.extract_taxon_subset --taxon "Orthocoronavirinae" --exact-match
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Set, Dict, Optional


# Default paths
DEFAULT_METADATA = "/mnt/data1/bfvd/metadata/bfvd_taxid_rank_scientificname_lineage.tsv"
DEFAULT_AA_FASTA = "/mnt/data1/bfvd/training_set/bfvd_data_aa.fasta"
DEFAULT_3DI_FASTA = "/mnt/data1/bfvd/training_set/bfvd_data_3di.fasta"
DEFAULT_3DI_MASKED_FASTA = "/mnt/data1/bfvd/training_set/bfvd_data_3di_masked.fasta"
DEFAULT_PLDDT_BINS_FASTA = "/mnt/data1/bfvd/training_set/bfvd_data_plddt_bins.fasta"


def parse_metadata_line(line: str) -> Optional[Dict]:
    """Parse a line from the metadata TSV file.
    
    Format: pdb_filename<TAB>taxid<TAB>rank<TAB>scientific_name<TAB>lineage
    """
    parts = line.strip().split('\t')
    if len(parts) < 5:
        return None
    
    pdb_filename = parts[0]
    # Extract the accession ID from the PDB filename
    # Format: A0A514CX81_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb
    accession = pdb_filename.split('_')[0] if '_' in pdb_filename else pdb_filename.replace('.pdb', '')
    
    return {
        'pdb_filename': pdb_filename,
        'accession': accession,
        'taxid': parts[1],
        'rank': parts[2],
        'scientific_name': parts[3],
        'lineage': parts[4] if len(parts) > 4 else ''
    }


def find_matching_accessions(
    metadata_path: str,
    taxon: str,
    exact_match: bool = False,
    search_field: str = 'all',
    verbose: bool = False
) -> Set[str]:
    """Find all accessions matching the given taxon query.
    
    Args:
        metadata_path: Path to the metadata TSV file
        taxon: Taxon name to search for
        exact_match: If True, require exact substring match (case-sensitive)
        search_field: Which field to search: 'all', 'name', 'lineage'
        verbose: Print matching info
        
    Returns:
        Set of accession IDs matching the query
    """
    matching_accessions = set()
    matching_taxa = set()
    
    if exact_match:
        pattern = re.compile(re.escape(taxon))
    else:
        pattern = re.compile(re.escape(taxon), re.IGNORECASE)
    
    with open(metadata_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            entry = parse_metadata_line(line)
            if entry is None:
                continue
            
            # Check which fields to search
            search_text = ''
            if search_field in ('all', 'name'):
                search_text += entry['scientific_name'] + ' '
            if search_field in ('all', 'lineage'):
                search_text += entry['lineage']
            
            if pattern.search(search_text):
                matching_accessions.add(entry['accession'])
                matching_taxa.add(entry['scientific_name'])
                
            if verbose and line_num % 100000 == 0:
                print(f"Processed {line_num:,} entries...", file=sys.stderr)
    
    if verbose:
        print(f"\nFound {len(matching_accessions):,} matching accessions", file=sys.stderr)
        print(f"From {len(matching_taxa):,} unique taxa", file=sys.stderr)
        if len(matching_taxa) <= 20:
            print("Matching taxa:", file=sys.stderr)
            for t in sorted(matching_taxa):
                print(f"  - {t}", file=sys.stderr)
    
    return matching_accessions


def read_fasta_to_dict(fasta_path: str, filter_accessions: Optional[Set[str]] = None) -> Dict[str, str]:
    """Read a FASTA file into a dictionary.
    
    Args:
        fasta_path: Path to FASTA file
        filter_accessions: If provided, only include sequences with these accessions
        
    Returns:
        Dict mapping header (without >) to sequence
    """
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                
                current_header = line[1:]  # Remove '>'
                current_seq = []
                
                # Skip if not in filter set
                if filter_accessions is not None:
                    accession = current_header.split()[0].split('_')[0]
                    if accession not in filter_accessions:
                        current_header = None
            elif current_header is not None:
                current_seq.append(line)
        
        # Save last sequence
        if current_header is not None:
            sequences[current_header] = ''.join(current_seq)
    
    return sequences


def extract_subset_streaming(
    input_fasta: str,
    output_fasta: str,
    filter_accessions: Set[str],
    verbose: bool = False
) -> int:
    """Extract subset from FASTA in streaming fashion (memory efficient).
    
    Returns:
        Number of sequences written
    """
    count = 0
    current_header = None
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


def list_taxa(metadata_path: str, level: str = 'family') -> None:
    """List available taxa at a given taxonomic level.
    
    Args:
        metadata_path: Path to metadata file
        level: Taxonomic level to list (domain, kingdom, phylum, class, order, family, genus, species)
    """
    level_prefixes = {
        'domain': 'd_',
        'kingdom': 'k_',
        'phylum': 'p_',
        'class': 'c_',
        'order': 'o_',
        'family': 'f_',
        'genus': 'g_',
        'species': 's_'
    }
    
    if level not in level_prefixes:
        print(f"Unknown level: {level}. Choose from: {', '.join(level_prefixes.keys())}", file=sys.stderr)
        return
    
    prefix = level_prefixes[level]
    taxa_counts = {}
    
    with open(metadata_path, 'r') as f:
        for line in f:
            entry = parse_metadata_line(line)
            if entry is None:
                continue
            
            lineage = entry['lineage']
            # Find the taxon at the requested level
            for part in lineage.split(';'):
                if part.startswith(prefix):
                    taxon = part[2:]  # Remove prefix
                    taxa_counts[taxon] = taxa_counts.get(taxon, 0) + 1
                    break
    
    # Sort by count
    sorted_taxa = sorted(taxa_counts.items(), key=lambda x: -x[1])
    
    print(f"\n{level.capitalize()}-level taxa ({len(sorted_taxa)} total):\n")
    for taxon, count in sorted_taxa:
        print(f"  {taxon}: {count:,} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Extract BFVD subset by taxonomic criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main arguments
    parser.add_argument('--taxon', '-t', type=str,
                        help='Taxon name to search for (searches in scientific name and lineage)')
    parser.add_argument('--output-prefix', '-o', type=str,
                        help='Output prefix for FASTA files (e.g., "coronaviridae" creates coronaviridae_aa.fasta)')
    
    # Path options
    parser.add_argument('--metadata', '-m', type=str, default=DEFAULT_METADATA,
                        help=f'Path to metadata TSV file (default: {DEFAULT_METADATA})')
    parser.add_argument('--aa-fasta', type=str, default=DEFAULT_AA_FASTA,
                        help=f'Path to amino acid FASTA (default: {DEFAULT_AA_FASTA})')
    parser.add_argument('--three-di-fasta', type=str, default=DEFAULT_3DI_FASTA,
                        help=f'Path to 3Di FASTA (default: {DEFAULT_3DI_FASTA})')
    parser.add_argument('--include-masked', action='store_true',
                        help='Also extract masked 3Di sequences')
    parser.add_argument('--masked-fasta', type=str, default=DEFAULT_3DI_MASKED_FASTA,
                        help=f'Path to masked 3Di FASTA (default: {DEFAULT_3DI_MASKED_FASTA})')
    parser.add_argument('--include-plddt-bins', action='store_true',
                        help='Also extract pLDDT bins FASTA (for pLDDT-weighted loss)')
    parser.add_argument('--plddt-bins-fasta', type=str, default=DEFAULT_PLDDT_BINS_FASTA,
                        help=f'Path to pLDDT bins FASTA (default: {DEFAULT_PLDDT_BINS_FASTA})')
    
    # Search options
    parser.add_argument('--exact-match', '-e', action='store_true',
                        help='Require exact (case-sensitive) match')
    parser.add_argument('--search-field', choices=['all', 'name', 'lineage'], default='all',
                        help='Which field to search (default: all)')
    
    # Alternative modes
    parser.add_argument('--list-taxa', type=str, metavar='LEVEL',
                        choices=['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'],
                        help='List available taxa at a given level instead of extracting')
    parser.add_argument('--count-only', action='store_true',
                        help='Only count matches, do not extract sequences')
    parser.add_argument('--accession-list', type=str,
                        help='Output file for list of matching accessions')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress information')
    parser.add_argument('--aa-only', action='store_true',
                        help='Only extract AA sequences (skip 3Di)')
    parser.add_argument('--three-di-only', action='store_true',
                        help='Only extract 3Di sequences (skip AA)')
    
    args = parser.parse_args()
    
    # Handle list-taxa mode
    if args.list_taxa:
        list_taxa(args.metadata, args.list_taxa)
        return
    
    # Require taxon for other modes
    if not args.taxon:
        parser.error("--taxon is required (unless using --list-taxa)")
    
    # Find matching accessions
    print(f"Searching for '{args.taxon}'...", file=sys.stderr)
    matching_accessions = find_matching_accessions(
        args.metadata,
        args.taxon,
        exact_match=args.exact_match,
        search_field=args.search_field,
        verbose=args.verbose
    )
    
    if not matching_accessions:
        print(f"No entries found matching '{args.taxon}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(matching_accessions):,} matching sequences", file=sys.stderr)
    
    # Count-only mode
    if args.count_only:
        return
    
    # Save accession list if requested
    if args.accession_list:
        with open(args.accession_list, 'w') as f:
            for acc in sorted(matching_accessions):
                f.write(f"{acc}\n")
        print(f"Saved accession list to {args.accession_list}", file=sys.stderr)
    
    # Require output prefix for extraction
    if not args.output_prefix:
        if not args.count_only and not args.accession_list:
            parser.error("--output-prefix is required for sequence extraction")
        return
    
    # Extract sequences
    output_dir = Path(args.output_prefix).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract AA sequences
    if not args.three_di_only:
        aa_output = f"{args.output_prefix}_aa.fasta"
        print(f"Extracting AA sequences to {aa_output}...", file=sys.stderr)
        extract_subset_streaming(args.aa_fasta, aa_output, matching_accessions, verbose=args.verbose)
    
    # Extract 3Di sequences
    if not args.aa_only:
        threedi_output = f"{args.output_prefix}_3di.fasta"
        print(f"Extracting 3Di sequences to {threedi_output}...", file=sys.stderr)
        extract_subset_streaming(args.three_di_fasta, threedi_output, matching_accessions, verbose=args.verbose)
        
        # Extract masked 3Di if requested
        if args.include_masked:
            masked_output = f"{args.output_prefix}_3di_masked.fasta"
            print(f"Extracting masked 3Di sequences to {masked_output}...", file=sys.stderr)
            extract_subset_streaming(args.masked_fasta, masked_output, matching_accessions, verbose=args.verbose)
    
    # Extract pLDDT bins if requested
    if args.include_plddt_bins:
        plddt_output = f"{args.output_prefix}_plddt_bins.fasta"
        print(f"Extracting pLDDT bins to {plddt_output}...", file=sys.stderr)
        extract_subset_streaming(args.plddt_bins_fasta, plddt_output, matching_accessions, verbose=args.verbose)
    
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()

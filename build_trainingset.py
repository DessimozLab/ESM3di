#!/usr/bin/env python
"""
Build training dataset from PDB files with pLDDT-based masking.

This script:
1. Reads a directory of PDB files
2. Uses FoldSeek to generate 3Di sequences
3. Extracts pLDDT scores from PDB B-factors
4. Masks low-confidence positions in 3Di sequences
5. Outputs amino acid and masked 3Di FASTA files for training

Requirements:
- FoldSeek installed and in PATH
- PDB files with pLDDT scores in B-factor column (e.g., AlphaFold predictions)
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


class PDBParser:
    """Simple PDB parser to extract sequences and pLDDT scores."""
    
    @staticmethod
    def parse_pdb(pdb_path: str) -> Dict[str, Tuple[str, List[float]]]:
        """
        Parse PDB file and extract sequence and pLDDT scores per chain.
        
        Returns:
            Dictionary mapping chain_id -> (sequence, plddt_scores)
        """
        chains = {}
        
        with open(pdb_path, 'r') as f:
            current_chain = None
            residues = {}  # chain -> {resnum: (aa, plddt)}
            
            for line in f:
                if line.startswith('ATOM  ') or line.startswith('HETATM'):
                    # Parse ATOM/HETATM line
                    atom_name = line[12:16].strip()
                    # Only use CA atoms for sequence
                    if atom_name != 'CA':
                        continue
                    
                    res_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    if not chain_id:
                        chain_id = 'A'  # Default chain
                    
                    res_num = line[22:26].strip()
                    insertion_code = line[26:27].strip()
                    res_id = f"{res_num}{insertion_code}"
                    
                    # B-factor column contains pLDDT
                    try:
                        plddt = float(line[60:66].strip())
                    except (ValueError, IndexError):
                        plddt = 0.0
                    
                    # Convert 3-letter to 1-letter amino acid code
                    aa = PDBParser.three_to_one(res_name)
                    
                    if chain_id not in residues:
                        residues[chain_id] = {}
                    
                    # Store by residue ID to handle insertions
                    if res_id not in residues[chain_id]:
                        residues[chain_id][res_id] = (aa, plddt)
        
        # Convert to sequences
        for chain_id, res_dict in residues.items():
            # Sort by residue number
            sorted_residues = sorted(
                res_dict.items(),
                key=lambda x: (int(re.match(r'-?\d+', x[0]).group()), x[0])
            )
            
            sequence = ''.join([aa for _, (aa, _) in sorted_residues])
            plddts = [plddt for _, (_, plddt) in sorted_residues]
            
            chains[chain_id] = (sequence, plddts)
        
        return chains
    
    @staticmethod
    def three_to_one(three_letter: str) -> str:
        """Convert 3-letter amino acid code to 1-letter."""
        conversion = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
            'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
            'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
            'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'SEC': 'U', 'PYL': 'O',  # Rare amino acids
        }
        return conversion.get(three_letter.upper(), 'X')


def check_foldseek_installed(foldseek_bin: str = "foldseek") -> bool:
    """Check if foldseek is available."""
    try:
        result = subprocess.run(
            [foldseek_bin, "version"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_foldseek_struct2profile(
    pdb_files: List[str],
    output_prefix: str,
    foldseek_bin: str = "foldseek"
) -> Tuple[str, str]:
    """
    Run FoldSeek to convert PDB structures to 3Di sequences.
    
    Returns:
        Tuple of (aa_fasta_path, three_di_fasta_path)
    """
    # Create temporary directory for FoldSeek database
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create list file with PDB paths
        list_file = os.path.join(tmpdir, "pdb_list.txt")
        with open(list_file, 'w') as f:
            for pdb_path in pdb_files:
                f.write(f"{os.path.abspath(pdb_path)}\n")
        
        # Create FoldSeek database from structures
        db_path = os.path.join(tmpdir, "struct_db")
        
        print(f"Creating FoldSeek database from {len(pdb_files)} PDB files...")
        cmd_createdb = [
            foldseek_bin,
            "createdb",
            list_file,
            db_path,
        ]
        
        result = subprocess.run(
            cmd_createdb,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Error creating database: {result.stderr}", file=sys.stderr)
            raise RuntimeError("FoldSeek createdb failed")
        
        # Convert to profile (generates 3Di sequences)
        print("Generating 3Di sequences...")
        
        # Use convert2fasta to get both AA and 3Di
        aa_fasta = f"{output_prefix}_aa.fasta"
        cmd_aa = [
            foldseek_bin,
            "convert2fasta",
            db_path,
            aa_fasta,
        ]
        
        result = subprocess.run(
            cmd_aa,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Error converting to AA FASTA: {result.stderr}", file=sys.stderr)
            raise RuntimeError("FoldSeek convert2fasta failed")
        
        # Get 3Di sequences from _ss database
        three_di_fasta = f"{output_prefix}_3di.fasta"
        cmd_3di = [
            foldseek_bin,
            "convert2fasta",
            db_path + "_ss",
            three_di_fasta,
        ]
        
        result = subprocess.run(
            cmd_3di,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Error converting to 3Di FASTA: {result.stderr}", file=sys.stderr)
            raise RuntimeError("FoldSeek convert2fasta failed for 3Di")
    
    return aa_fasta, three_di_fasta


def read_fasta(fasta_path: str) -> Dict[str, str]:
    """Read FASTA file and return dict of header -> sequence."""
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_header is not None:
            sequences[current_header] = ''.join(current_seq)
    
    return sequences


def write_fasta(sequences: Dict[str, str], output_path: str):
    """Write sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for header, seq in sequences.items():
            f.write(f">{header}\n")
            # Write in lines of 80 characters
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def mask_3di_by_plddt(
    three_di_seq: str,
    plddt_scores: List[float],
    plddt_threshold: float,
    mask_char: str = 'X'
) -> str:
    """
    Mask 3Di sequence positions with low pLDDT scores.
    
    Args:
        three_di_seq: 3Di sequence string
        plddt_scores: List of pLDDT scores (one per residue)
        plddt_threshold: Threshold below which to mask (e.g., 70)
        mask_char: Character to use for masking
        
    Returns:
        Masked 3Di sequence
    """
    if len(three_di_seq) != len(plddt_scores):
        warnings.warn(
            f"Length mismatch: 3Di={len(three_di_seq)}, pLDDT={len(plddt_scores)}"
        )
        # Use minimum length
        min_len = min(len(three_di_seq), len(plddt_scores))
        three_di_seq = three_di_seq[:min_len]
        plddt_scores = plddt_scores[:min_len]
    
    masked_seq = []
    for three_di_char, plddt in zip(three_di_seq, plddt_scores):
        if plddt < plddt_threshold:
            masked_seq.append(mask_char)
        else:
            masked_seq.append(three_di_char)
    
    return ''.join(masked_seq)


def extract_structure_id(pdb_path: str, chain: str = None) -> str:
    """Extract structure identifier from PDB filename."""
    basename = os.path.basename(pdb_path)
    name = os.path.splitext(basename)[0]
    
    if chain:
        return f"{name}_{chain}"
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Build training dataset from PDB files with pLDDT-based 3Di masking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory of AlphaFold PDB files
  python build_trainingset.py \\
      --pdb-dir alphafold_structures/ \\
      --output-prefix training_data \\
      --plddt-threshold 70 \\
      --mask-char X
  
  # Process specific PDB files
  python build_trainingset.py \\
      --pdb-files structure1.pdb structure2.pdb structure3.pdb \\
      --output-prefix my_dataset \\
      --plddt-threshold 50
  
  # Use custom FoldSeek binary
  python build_trainingset.py \\
      --pdb-dir structures/ \\
      --output-prefix data \\
      --foldseek-bin /path/to/foldseek

Output files:
  - {prefix}_aa.fasta: Amino acid sequences
  - {prefix}_3di_masked.fasta: 3Di sequences with low-pLDDT positions masked
  - {prefix}_stats.txt: Statistics about masking
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdb-dir",
        type=str,
        help="Directory containing PDB files"
    )
    input_group.add_argument(
        "--pdb-files",
        type=str,
        nargs='+',
        help="List of PDB files to process"
    )
    
    # Output options
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Prefix for output files"
    )
    
    # Masking options
    parser.add_argument(
        "--plddt-threshold",
        type=float,
        default=70.0,
        help="pLDDT threshold below which to mask (default: 70)"
    )
    parser.add_argument(
        "--mask-char",
        type=str,
        default='X',
        help="Character to use for masking low-confidence positions (default: X)"
    )
    
    # Chain handling
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Specific chain to extract (default: all chains)"
    )
    parser.add_argument(
        "--split-chains",
        action="store_true",
        help="Split multi-chain structures into separate entries"
    )
    
    # FoldSeek options
    parser.add_argument(
        "--foldseek-bin",
        type=str,
        default="foldseek",
        help="Path to foldseek binary (default: foldseek in PATH)"
    )
    parser.add_argument(
        "--skip-foldseek",
        action="store_true",
        help="Skip FoldSeek (use when you have pre-generated 3Di FASTA)"
    )
    parser.add_argument(
        "--three-di-fasta",
        type=str,
        default=None,
        help="Pre-generated 3Di FASTA file (requires --skip-foldseek)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.skip_foldseek and not args.three_di_fasta:
        parser.error("--three-di-fasta is required when --skip-foldseek is set")
    
    if len(args.mask_char) != 1:
        parser.error("--mask-char must be a single character")
    
    # Get list of PDB files
    if args.pdb_dir:
        pdb_dir = Path(args.pdb_dir)
        if not pdb_dir.exists():
            print(f"Error: Directory not found: {args.pdb_dir}", file=sys.stderr)
            sys.exit(1)
        
        pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.ent"))
        if not pdb_files:
            print(f"Error: No PDB files found in {args.pdb_dir}", file=sys.stderr)
            sys.exit(1)
        
        pdb_files = [str(f) for f in pdb_files]
    else:
        pdb_files = args.pdb_files
        for pdb_file in pdb_files:
            if not os.path.exists(pdb_file):
                print(f"Error: File not found: {pdb_file}", file=sys.stderr)
                sys.exit(1)
    
    print(f"Found {len(pdb_files)} PDB files")
    
    # Check FoldSeek unless skipping
    if not args.skip_foldseek:
        print("Checking for FoldSeek installation...")
        if not check_foldseek_installed(args.foldseek_bin):
            print(
                "ERROR: FoldSeek not found. Please install FoldSeek first.",
                file=sys.stderr
            )
            print("Download from: https://github.com/steineggerlab/foldseek", file=sys.stderr)
            sys.exit(1)
        print("✓ FoldSeek found")
    
    # Step 1: Parse PDB files to extract pLDDT scores
    print("\nParsing PDB files to extract sequences and pLDDT scores...")
    pdb_data = {}  # structure_id -> {chain -> (sequence, plddts)}
    
    for pdb_file in pdb_files:
        try:
            chains = PDBParser.parse_pdb(pdb_file)
            struct_id = extract_structure_id(pdb_file)
            
            if args.chain:
                # Only keep specified chain
                if args.chain in chains:
                    pdb_data[struct_id] = {args.chain: chains[args.chain]}
                else:
                    print(f"Warning: Chain {args.chain} not found in {pdb_file}")
            else:
                pdb_data[struct_id] = chains
                
        except Exception as e:
            print(f"Warning: Error parsing {pdb_file}: {e}", file=sys.stderr)
    
    total_chains = sum(len(chains) for chains in pdb_data.values())
    print(f"✓ Parsed {len(pdb_data)} structures with {total_chains} chains")
    
    # Step 2: Generate 3Di sequences with FoldSeek
    if args.skip_foldseek:
        print(f"\nSkipping FoldSeek, using provided 3Di FASTA: {args.three_di_fasta}")
        three_di_fasta = args.three_di_fasta
        # Generate AA FASTA from PDB data
        aa_fasta = f"{args.output_prefix}_aa.fasta"
        aa_sequences = {}
        for struct_id, chains in pdb_data.items():
            for chain_id, (seq, _) in chains.items():
                if args.split_chains:
                    header = f"{struct_id}_{chain_id}"
                else:
                    header = struct_id
                aa_sequences[header] = seq
        write_fasta(aa_sequences, aa_fasta)
        print(f"✓ Wrote AA FASTA to {aa_fasta}")
    else:
        aa_fasta, three_di_fasta = run_foldseek_struct2profile(
            pdb_files,
            args.output_prefix,
            args.foldseek_bin
        )
        print(f"✓ Generated 3Di sequences")
        print(f"  - AA FASTA: {aa_fasta}")
        print(f"  - 3Di FASTA: {three_di_fasta}")
    
    # Step 3: Read 3Di sequences
    print("\nReading 3Di sequences...")
    three_di_sequences = read_fasta(three_di_fasta)
    print(f"✓ Read {len(three_di_sequences)} 3Di sequences")
    
    # Step 4: Mask 3Di sequences based on pLDDT
    print(f"\nMasking 3Di sequences (threshold: {args.plddt_threshold})...")
    masked_sequences = {}
    stats = {
        'total_residues': 0,
        'masked_residues': 0,
        'sequences_processed': 0,
        'sequences_skipped': 0,
    }
    
    for header, three_di_seq in three_di_sequences.items():
        # Find corresponding pLDDT data
        # FoldSeek headers might be formatted differently
        found = False
        
        for struct_id, chains in pdb_data.items():
            for chain_id, (aa_seq, plddts) in chains.items():
                # Try various header formats
                possible_headers = [
                    struct_id,
                    f"{struct_id}_{chain_id}",
                    f"{struct_id} {chain_id}",
                ]
                
                if any(h in header for h in possible_headers):
                    # Mask the sequence
                    masked_seq = mask_3di_by_plddt(
                        three_di_seq,
                        plddts,
                        args.plddt_threshold,
                        args.mask_char
                    )
                    
                    masked_sequences[header] = masked_seq
                    
                    # Update stats
                    stats['total_residues'] += len(masked_seq)
                    stats['masked_residues'] += masked_seq.count(args.mask_char)
                    stats['sequences_processed'] += 1
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            print(f"Warning: No pLDDT data found for {header}, skipping masking")
            masked_sequences[header] = three_di_seq
            stats['sequences_skipped'] += 1
    
    # Step 5: Write masked 3Di FASTA
    output_3di_masked = f"{args.output_prefix}_3di_masked.fasta"
    write_fasta(masked_sequences, output_3di_masked)
    print(f"✓ Wrote masked 3Di FASTA to {output_3di_masked}")
    
    # Step 6: Write statistics
    stats_file = f"{args.output_prefix}_stats.txt"
    mask_percentage = (stats['masked_residues'] / stats['total_residues'] * 100
                      if stats['total_residues'] > 0 else 0)
    
    with open(stats_file, 'w') as f:
        f.write("Training Dataset Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"PDB files processed: {len(pdb_files)}\n")
        f.write(f"Sequences processed: {stats['sequences_processed']}\n")
        f.write(f"Sequences skipped: {stats['sequences_skipped']}\n")
        f.write(f"Total residues: {stats['total_residues']}\n")
        f.write(f"Masked residues: {stats['masked_residues']}\n")
        f.write(f"Mask percentage: {mask_percentage:.2f}%\n")
        f.write(f"pLDDT threshold: {args.plddt_threshold}\n")
        f.write(f"Mask character: '{args.mask_char}'\n")
    
    print(f"\n✓ Wrote statistics to {stats_file}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Sequences processed: {stats['sequences_processed']}")
    print(f"Total residues: {stats['total_residues']}")
    print(f"Masked residues: {stats['masked_residues']} ({mask_percentage:.2f}%)")
    print(f"\nOutput files:")
    print(f"  - {aa_fasta}")
    print(f"  - {output_3di_masked}")
    print(f"  - {stats_file}")
    print("\n✓ Dataset ready for training!")


if __name__ == "__main__":
    main()

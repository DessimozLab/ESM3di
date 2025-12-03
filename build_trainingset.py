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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pebble import ProcessPool
import multiprocessing as mp


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



def read_dbfiles3di(  AADB , threeDidb):
    #find positions 
    threeDiseq = [ l.strip().replace('\x00','') for l in open(threeDidb)]
    lookup = AADB+'.lookup'
    ids = [ l.split()[1].strip().replace('.pdb', '').split('/')[-1] for l in open(lookup)]
    AAs = [ l.strip().replace('\x00','') for l in open(AADB)]
    mapper3di = dict(zip(ids,threeDiseq))
    mapperAA = dict(zip(ids,AAs))
    return mapper3di, mapperAA

def mapper2fasta(mapper3di, mapperAA, output_prefix):
    """
    Write the 3Di and AA sequences to FASTA files.
    
    Args:
        mapper3di: Dictionary mapping IDs to 3Di sequences
        mapperAA: Dictionary mapping IDs to AA sequences
        output_prefix: Prefix for output files
    """

    aa_fasta_path = f"{output_prefix}_aa.fasta"
    three_di_fasta_path = f"{output_prefix}_3di.fasta"
    
    # Write AA sequences
    with open(aa_fasta_path, 'w') as aa_fasta:
        for id, seq in tqdm(mapperAA.items(), desc="Writing AA sequences", unit="seq"):
            aa_fasta.write(f">{id}\n{seq}\n")
    
    # Write 3Di sequences
    with open(three_di_fasta_path, 'w') as three_di_fasta:
        for id, seq in tqdm(mapper3di.items(), desc="Writing 3Di sequences", unit="seq"):
            three_di_fasta.write(f">{id}\n{seq}\n")
    
    return aa_fasta_path, three_di_fasta_path


def run_foldseek_struct2profile(
    structure_dir: str,
    output_prefix: str,
    foldseek_bin: str = "foldseek"
) -> Tuple[str, str]:
    """
    Run FoldSeek to convert PDB structures to 3Di sequences.
    
    Args:
        structure_dir: Directory containing PDB structure files
        output_prefix: Prefix for output files
        foldseek_bin: Path to foldseek binary
    
    Returns:
        Tuple of (aa_fasta_path, three_di_fasta_path)
    """
    # Create FoldSeek database from structure directory
    db_path = f"{output_prefix}_db"
    
    print(f"Creating FoldSeek database from structures in {structure_dir}...")
    cmd_createdb = [
        foldseek_bin,
        "createdb",
        structure_dir,
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
    print("Generating fasta files...")
    mapper3di, mapperAA = read_dbfiles3di(db_path , db_path + '_ss')
    aa_fasta, three_di_fasta = mapper2fasta(mapper3di, mapperAA, output_prefix)
    print(f"✓ Generated AA FASTA: {aa_fasta}")
    print(f"✓ Generated 3Di FASTA: {three_di_fasta}")
    return aa_fasta, three_di_fasta


def read_fasta(fasta_path: str, show_progress: bool = False) -> Dict[str, str]:
    """Read FASTA file and return dict of header -> sequence."""
    sequences = {}
    current_header = None
    current_seq = []
    
    # Count lines for progress bar
    if show_progress:
        with open(fasta_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        file_iter = open(fasta_path, 'r')
        line_iter = tqdm(file_iter, total=total_lines, desc="Reading FASTA", unit="line")
    else:
        file_iter = open(fasta_path, 'r')
        line_iter = file_iter
    
    with file_iter:
        for line in line_iter:
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


def write_fasta(sequences: Dict[str, str], output_path: str , order: Optional[List[str]] = None):
    """Write sequences to FASTA file."""
    if order is not None:
        sequences = {k: sequences[k] for k in order if k in sequences}
    
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
    masked_seq = [ mask_char if plddt < plddt_threshold else three_di_char for three_di_char, plddt in zip(three_di_seq, plddt_scores) ]
    return ''.join(masked_seq)  
    

def mask_sequence_worker(header, three_di_seq, pdb_data, plddt_threshold, mask_char):
    """Worker function to mask a single sequence."""    
    # Find corresponding pLDDT data
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
                    plddt_threshold,
                    mask_char
                )
                
                # Return results for stats
                return header, masked_seq, len(masked_seq), masked_seq.count(mask_char), True
        
            if found:
                break
    
    # No pLDDT data found
    return header, three_di_seq, 0, 0, False

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
  # FoldSeek will use the directory directly to create a database
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

Note:
  FoldSeek creates a database directly from the folder of structures,
  no temporary directories or PDB lists are created.

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
    
    for pdb_file in tqdm(pdb_files, desc="Parsing PDB files", unit="file"):
        try:
            chains = PDBParser.parse_pdb(pdb_file)
            struct_id = extract_structure_id(pdb_file)
            
            if args.chain:
                # Only keep specified chain
                if args.chain in chains:
                    pdb_data[struct_id] = {args.chain: chains[args.chain]}
                else:
                    tqdm.write(f"Warning: Chain {args.chain} not found in {pdb_file}")
            else:
                pdb_data[struct_id] = chains
                
        except Exception as e:
            tqdm.write(f"Warning: Error parsing {pdb_file}: {e}")
    
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
        # Use the directory containing structures
        if args.pdb_dir:
            structure_dir = args.pdb_dir
        else:
            # If individual files were provided, use parent directory of first file
            structure_dir = os.path.dirname(os.path.abspath(pdb_files[0]))
            print(f"Using structure directory: {structure_dir}")
        
        aa_fasta, three_di_fasta = run_foldseek_struct2profile(
            structure_dir,
            args.output_prefix,
            args.foldseek_bin
        )
        print(f"✓ Generated 3Di sequences")
        print(f"  - AA FASTA: {aa_fasta}")
        print(f"  - 3Di FASTA: {three_di_fasta}")
    
    # Step 3: Read 3Di sequences
    print("\nReading 3Di sequences...")
    three_di_sequences = read_fasta(three_di_fasta, show_progress=True)
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
    
   
    # Prepare arguments for multiprocessing
    # Prepare arguments for multiprocessing using dictionary and kwargs
    mask_args = []
    for header, three_di_seq in three_di_sequences.items():
        mask_args.append({
            'header': header,
            'three_di_seq': three_di_seq,
            'pdb_data': { header: pdb_data[header] },
            'plddt_threshold': args.plddt_threshold,
            'mask_char': args.mask_char
        })
    
    # Use pebble ProcessPool for robust multiprocessing
    num_workers = min(mp.cpu_count(), len(mask_args))
    
    with ProcessPool(max_workers=num_workers) as pool:
        # Submit all tasks using kwargs
        future_to_args = {
            pool.schedule(mask_sequence_worker, kwargs=arg_dict): arg_dict['header'] 
            for arg_dict in mask_args
        }
        
        # Process results with progress bar
        with tqdm(total=len(mask_args), desc="Masking sequences", unit="seq") as pbar:
            for future in as_completed(future_to_args):
                try:
                    header, masked_seq, total_res, masked_res, found = future.result()
                    
                    masked_sequences[header] = masked_seq
                    
                    if found:
                        stats['total_residues'] += total_res
                        stats['masked_residues'] += masked_res
                        stats['sequences_processed'] += 1
                    else:
                        tqdm.write(f"Warning: No pLDDT data found for {header}, skipping masking")
                        stats['sequences_skipped'] += 1
                        
                except Exception as e:
                    header = future_to_args[future]
                    tqdm.write(f"Error processing {header}: {e}")
                    stats['sequences_skipped'] += 1
                pbar.update(1)
    
    # Step 5: Write masked 3Di FASTA
    output_3di_masked = f"{args.output_prefix}_3di_masked.fasta"
    write_fasta(masked_sequences, output_3di_masked , order= list(three_di_sequences.keys()))
    
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

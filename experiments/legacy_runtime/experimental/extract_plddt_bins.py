#!/usr/bin/env python
"""
Extract pLDDT bins from PDB files and output as FASTA.
Matches the order of an existing AA FASTA file.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def plddt_to_bin(plddt: float) -> int:
    """Convert pLDDT score (0-100) to bin (0-9)."""
    bin_idx = int(plddt // 10)
    return min(bin_idx, 9)


def parse_pdb_plddt(pdb_path: str) -> Tuple[str, str]:
    """
    Parse PDB file and extract sequence and pLDDT bins.
    Returns (aa_sequence, plddt_bins_string)
    """
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U', 'PYL': 'O',
    }
    
    residues = {}  # resnum -> (aa, plddt)
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM  ') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                insertion = line[26:27].strip()
                res_id = f"{res_num}{insertion}"
                
                try:
                    plddt = float(line[60:66].strip())
                except (ValueError, IndexError):
                    plddt = 0.0
                
                aa = three_to_one.get(res_name.upper(), 'X')
                
                if res_id not in residues:
                    residues[res_id] = (aa, plddt)
    
    # Sort by residue number
    import re
    sorted_residues = sorted(
        residues.items(),
        key=lambda x: (int(re.match(r'-?\d+', x[0]).group()), x[0])
    )
    
    sequence = ''.join(aa for _, (aa, _) in sorted_residues)
    plddt_bins = ''.join(str(plddt_to_bin(plddt)) for _, (_, plddt) in sorted_residues)
    
    return sequence, plddt_bins


def read_fasta_headers(fasta_path: str) -> List[str]:
    """Read FASTA and return list of headers in order."""
    headers = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                headers.append(line[1:].strip())
    return headers


def main():
    parser = argparse.ArgumentParser(
        description="Extract pLDDT bins from PDB files to FASTA"
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        required=True,
        help="Directory containing PDB files"
    )
    parser.add_argument(
        "--aa-fasta",
        type=str,
        required=True,
        help="Reference AA FASTA to match order (headers should match PDB names)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pLDDT bins FASTA file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Read reference FASTA headers
    print(f"Reading reference FASTA: {args.aa_fasta}")
    headers = read_fasta_headers(args.aa_fasta)
    print(f"Found {len(headers)} sequences")
    
    # Process PDB files
    pdb_dir = Path(args.pdb_dir)
    
    print(f"Extracting pLDDT bins from PDB files in {args.pdb_dir}...")
    
    results = {}
    missing = []
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Build list of PDB files to process
    pdb_tasks = []
    for header in headers:
        # Try different PDB naming conventions
        possible_names = [
            f"{header}.pdb",
            f"{header.split()[0]}.pdb",
            f"{header.split('_')[0]}.pdb",
        ]
        
        pdb_path = None
        for name in possible_names:
            candidate = pdb_dir / name
            if candidate.exists():
                pdb_path = candidate
                break
        
        if pdb_path:
            pdb_tasks.append((header, str(pdb_path)))
        else:
            missing.append(header)
    
    if missing:
        print(f"Warning: {len(missing)} PDB files not found")
        if len(missing) <= 10:
            for h in missing:
                print(f"  - {h}")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(parse_pdb_plddt, pdb_path): header
            for header, pdb_path in pdb_tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Extracting pLDDT", unit="pdb"):
            header = futures[future]
            try:
                _, plddt_bins = future.result()
                results[header] = plddt_bins
            except Exception as e:
                print(f"Error processing {header}: {e}")
    
    # Write output FASTA in original order
    print(f"Writing output to {args.output}")
    with open(args.output, 'w') as f:
        for header in headers:
            if header in results:
                f.write(f">{header}\n")
                bins = results[header]
                for i in range(0, len(bins), 80):
                    f.write(bins[i:i+80] + "\n")
    
    print(f"✓ Wrote {len(results)} sequences to {args.output}")
    if missing:
        print(f"  ({len(missing)} sequences skipped - PDB not found)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Tree building and FoldSeek utilities for comparative phylogenetic analysis.

This module provides functions for:
- Running FoldSeek for structure-based sequence extraction
- Multiple sequence alignment with MAFFT
- Tree building with RAxML-ng and QuickTree
- Distance matrix handling
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from Bio import SeqIO


# -----------------------------
# FoldSeek Functions
# -----------------------------

def check_foldseek_installed(foldseek_bin: str = "foldseek") -> bool:
    """Check if foldseek is available in PATH."""
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


def run_foldseek_createdb(
    input_folder: str,
    output_folder: str,
    db_name: str = "structdb",
    foldseek_bin: str = "foldseek"
) -> str:
    """
    Run foldseek createdb to create a structure database.
    
    Args:
        input_folder: Path to folder with PDB/mmCIF files
        output_folder: Path to output folder
        db_name: Name of the database
        foldseek_bin: Path to foldseek executable
    
    Returns:
        Path to the created database
    """
    os.makedirs(output_folder, exist_ok=True)
    db_path = os.path.join(output_folder, db_name)
    
    cmd = [foldseek_bin, "createdb", input_folder, db_path]
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Foldseek createdb failed: {result.stderr}")
    
    print(f"✓ FoldSeek database created at {db_path}")
    return db_path


def run_foldseek_allvall(
    input_folder: str,
    output_path: str,
    foldseek_bin: str = "foldseek",
    tmp_dir: str = "tmp"
) -> str:
    """
    Run foldseek easy-search for all-vs-all structure comparison.
    
    Args:
        input_folder: Path to folder with structure files
        output_path: Path for output results
        foldseek_bin: Path to foldseek executable
        tmp_dir: Temporary directory for foldseek
    
    Returns:
        Path to output file
    """
    cmd = [
        foldseek_bin, "easy-search",
        input_folder, input_folder, output_path, tmp_dir,
        "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits",
        "--exhaustive-search"
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Foldseek easy-search failed: {result.stderr}")
    
    print(f"✓ Foldseek all vs all completed: {output_path}")
    return output_path


def read_foldseek_db(db: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Read 3Di and amino acid sequences from FoldSeek database files.
    
    Args:
        db: Path to FoldSeek database
        name: Database name prefix
        name: Database name prefix
    
    Returns:
        Tuple of (3di_sequences_dict, aa_sequences_dict)
    """
    folder = os.path.dirname(db)
    name = os.path.basename(db)
    
    # Read 3Di sequences
    threedi_db = os.path.join(folder, f"{name}_ss")
    threedi_seqs = [l.strip().replace('\x00', '') for l in open(threedi_db)]
    
    # Read lookup table for IDs
    lookup = os.path.join(folder, f"{name}.lookup")
    ids = [l.split()[1].strip() for l in open(lookup)]
    
    # Read amino acid sequences
    aa_db = os.path.join(folder, name)
    aa_seqs = [l.strip().replace('\x00', '') for l in open(aa_db)]
    
    mapper_3di = dict(zip(ids, threedi_seqs))
    mapper_aa = dict(zip(ids, aa_seqs))
    
    return mapper_3di, mapper_aa


# -----------------------------
# Distance Matrix Functions
# -----------------------------

def tajima_distance(kn_ratio: np.ndarray, bfactor: float = 0.93, iterations: int = 100) -> np.ndarray:
    """
    Calculate Tajima distance from k/n ratio matrix.
    
    Args:
        kn_ratio: Matrix of k/n ratios
        bfactor: B factor parameter (default: 0.93)
        iterations: Number of iterations for summation
    
    Returns:
        Distance matrix with diagonal set to 0
    """
    taj = np.add.reduce([
        (kn_ratio ** (np.ones(kn_ratio.shape) * i)) / (bfactor ** (i - 1) * i)
        for i in range(1, iterations)
    ])
    np.fill_diagonal(taj, 0)
    return taj


def write_distance_matrix(
    identifiers: List[str],
    distmat: np.ndarray,
    output_file: str
) -> str:
    """
    Write distance matrix in PHYLIP/FastME format.
    
    Args:
        identifiers: List of sequence identifiers
        distmat: Distance matrix (numpy array)
        output_file: Path to output file
    
    Returns:
        Path to output file
    """
    outstr = f"{len(identifiers)}\n"
    for i, name in enumerate(identifiers):
        distances = ' '.join([f"{d:.4f}" for d in distmat[i, :]])
        outstr += f"{name} {distances}\n"
    
    with open(output_file, 'w') as f:
        f.write(outstr)
    
    print(f"✓ Distance matrix written to {output_file}")
    return output_file


# -----------------------------
# Alignment Functions
# -----------------------------

def run_mafft(
    input_fasta: str,
    output_fasta: str,
    algorithm: str = "auto",
    mafft_bin: str = "mafft",
    threads: int = 4
) -> str:
    """
    Run MAFFT multiple sequence alignment.
    
    Args:
        input_fasta: Path to input FASTA file
        output_fasta: Path to output aligned FASTA
        algorithm: MAFFT algorithm ('auto', 'linsi', 'ginsi', 'einsi', 'fftns')
        mafft_bin: Path to MAFFT binary
        threads: Number of threads
    
    Returns:
        Path to aligned FASTA file
    """
    print(f"Running MAFFT alignment ({algorithm})...")
    
    cmd = [mafft_bin, "--thread", str(threads)]
    
    # Add algorithm-specific flags
    if algorithm == "linsi":
        cmd.extend(["--localpair", "--maxiterate", "1000"])
    elif algorithm == "ginsi":
        cmd.extend(["--globalpair", "--maxiterate", "1000"])
    elif algorithm == "einsi":
        cmd.extend(["--genafpair", "--maxiterate", "1000"])
    elif algorithm == "fftns":
        cmd.extend(["--retree", "2", "--maxiterate", "0"])
    elif algorithm != "auto":
        print(f"Warning: Unknown algorithm '{algorithm}', using auto")
    
    cmd.append(input_fasta)
    
    # Run MAFFT
    with open(output_fasta, 'w') as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr}")
    
    # Count sequences in output
    num_seqs = sum(1 for line in open(output_fasta) if line.startswith('>'))
    print(f"✓ Aligned {num_seqs} sequences -> {output_fasta}")
    
    return output_fasta


# -----------------------------
# Tree Building Functions
# -----------------------------

def run_raxml(
    alignment_fasta: str,
    output_prefix: str,
    model: str = "LG+G4",
    raxml_bin: str = "raxml-ng",
    threads: int = 4,
    bootstrap: int = 0,
    seed: int = 12345
) -> str:
    """
    Build maximum likelihood tree using RAxML-ng.
    
    Args:
        alignment_fasta: Path to aligned FASTA file
        output_prefix: Prefix for output files
        model: Substitution model (e.g., 'LG+G4', 'WAG+I+G4', 'PROTGTR+G4')
        raxml_bin: Path to RAxML-ng binary
        threads: Number of threads
        bootstrap: Number of bootstrap replicates (0 = no bootstrap)
        seed: Random seed
    
    Returns:
        Path to best tree file (Newick format)
    """
    print(f"Running RAxML-ng ({model})...")
    
    cmd = [
        raxml_bin,
        "--msa", alignment_fasta,
        "--model", model,
        "--prefix", output_prefix,
        "--threads", str(threads),
        "--seed", str(seed),
        "--redo"  # Overwrite existing files
    ]
    
    if bootstrap > 0:
        cmd.extend(["--all", "--bs-trees", str(bootstrap)])
    else:
        cmd.append("--search")
    
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"RAxML stderr: {result.stderr}")
        raise RuntimeError(f"RAxML-ng failed")
    
    # Find output tree file
    tree_file = f"{output_prefix}.raxml.bestTree"
    if bootstrap > 0:
        support_tree = f"{output_prefix}.raxml.support"
        if os.path.exists(support_tree):
            tree_file = support_tree
    
    print(f"✓ Tree saved to {tree_file}")
    return tree_file


def run_quicktree(
    distmat_file: str,
    output_file: str,
    quicktree_bin: str = "quicktree"
) -> str:
    """
    Run QuickTree for distance-based tree building.
    
    Args:
        distmat_file: Path to distance matrix in PHYLIP format
        output_file: Path for output tree (Newick format)
        quicktree_bin: Path to quicktree binary
    
    Returns:
        Path to output tree file
    """
    print(f"Running QuickTree...")
    
    cmd = f"{quicktree_bin} -i m {distmat_file}"
    print(f"  Command: {cmd}")
    
    with open(output_file, 'w') as out_f:
        result = subprocess.run(
            cmd.split(),
            stdout=out_f,
            stderr=subprocess.PIPE,
            text=True
        )
    
    if result.returncode != 0:
        raise RuntimeError(f"QuickTree failed: {result.stderr}")
    
    print(f"✓ Tree saved to {output_file}")
    return output_file


# -----------------------------
# 3Di Prediction
# -----------------------------

def predict_3di_from_sequences(
    input_fasta: str,
    output_fasta: str,
    checkpoint_path: str,
    hf_model_name: str = "facebook/esm2_t12_35M_UR50D",
    batch_size: int = 4,
    device: str = None
) -> str:
    """
    Predict 3Di sequences from amino acid sequences using ESM3Di model.
    This does NOT use structure information, only the sequence.
    
    Args:
        input_fasta: Path to input amino acid FASTA
        output_fasta: Path to output 3Di FASTA
        checkpoint_path: Path to ESM3Di model checkpoint
        hf_model_name: HuggingFace model name
        batch_size: Inference batch size
        device: Device ('cuda' or 'cpu')
    
    Returns:
        Path to output FASTA file
    """
    import torch
    from .ESM3di_model import ESM3DiModel
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading ESM3Di model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    args_dict = checkpoint.get('args', {})
    num_labels = len(checkpoint.get('label_vocab', []))
    use_cnn_head = args_dict.get('use_cnn_head', False)
    use_transformer_head = args_dict.get('use_transformer_head', False)
    lora_r = args_dict.get('lora_r', 8)
    lora_alpha = args_dict.get('lora_alpha', 16)
    lora_dropout = args_dict.get('lora_dropout', 0.05)
    
    # Use model name from checkpoint if available
    if 'hf_model_name' in args_dict:
        hf_model_name = args_dict['hf_model_name']
    
    print(f"  Base model: {hf_model_name}")
    print(f"  Labels: {num_labels}")
    print(f"  Device: {device}")
    
    # Initialize model
    model = ESM3DiModel(
        hf_model_name=hf_model_name,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_cnn_head=use_cnn_head,
        cnn_num_layers=args_dict.get('cnn_num_layers', 2),
        cnn_kernel_size=args_dict.get('cnn_kernel_size', 3),
        cnn_dropout=args_dict.get('cnn_dropout', 0.1),
        use_transformer_head=use_transformer_head,
        transformer_head_dim=args_dict.get('transformer_head_dim', 256),
        transformer_head_layers=args_dict.get('transformer_head_layers', 2),
        transformer_head_dropout=args_dict.get('transformer_head_dropout', 0.1),
        transformer_head_num_heads=args_dict.get('transformer_head_num_heads', None),
    )
    
    # Run inference
    print(f"\nPredicting 3Di sequences...")
    model.predict_from_fasta(
        input_fasta_path=input_fasta,
        output_fasta_path=output_fasta,
        model_checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device
    )
    
    print(f"\n✓ Predicted 3Di written to {output_fasta}")
    return output_fasta

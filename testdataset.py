#!/usr/bin/env python
"""
Download random sample of PDB structures from AlphaFold Database.

This script downloads a specified number of random AlphaFold predicted structures
from the AlphaFold Protein Structure Database. Structures can be filtered by
organism and other criteria.

AlphaFold Database: https://alphafold.ebi.ac.uk/
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict
import urllib.request
import urllib.error


class AlphaFoldDownloader:
    """Download structures from AlphaFold Database."""
    
    # AlphaFold DB base URLs
    BASE_URL = "https://alphafold.ebi.ac.uk"
    API_URL = "https://alphafold.ebi.ac.uk/api"
    
    # Common model organisms
    ORGANISMS = {
        'human': 'Homo sapiens',
        'mouse': 'Mus musculus',
        'ecoli': 'Escherichia coli',
        'yeast': 'Saccharomyces cerevisiae',
        'arabidopsis': 'Arabidopsis thaliana',
        'celegans': 'Caenorhabditis elegans',
        'drosophila': 'Drosophila melanogaster',
        'zebrafish': 'Danio rerio',
    }
    
    # Pre-computed lists of accessions by organism (sample)
    # Note: For production use, you'd query the API or use a full list
    SAMPLE_ACCESSIONS = {
        'human': [
            'P04637', 'P53_HUMAN',  # p53
            'P01112', 'RASH_HUMAN',  # H-Ras
            'P42574', 'CASP3_HUMAN',  # Caspase-3
        ],
    }
    
    def __init__(self, output_dir: str, delay: float = 0.5):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded structures
            delay: Delay between downloads in seconds (be nice to servers)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
    
    @staticmethod
    def get_uniprot_list(organism: Optional[str] = None, 
                         reviewed: bool = False) -> List[str]:
        """
        Get list of UniProt accessions (simplified approach).
        
        For a real implementation, you'd query UniProt API:
        https://www.uniprot.org/help/api_queries
        """
        # This is a placeholder - in practice, you'd query UniProt API
        # or use a pre-downloaded list of accessions
        print("Note: Using sample accessions. For full coverage, query UniProt API.")
        
        if organism and organism.lower() in AlphaFoldDownloader.SAMPLE_ACCESSIONS:
            return AlphaFoldDownloader.SAMPLE_ACCESSIONS[organism.lower()]
        
        # Return some example accessions
        return [
            'Q5VSL9', 'Q8N6T3', 'Q9Y6K9', 'P62258', 'P68431',
            'Q96GQ7', 'O14976', 'Q9UKV8', 'P31946', 'P04637',
            'O15355', 'Q13485', 'P42574', 'Q15796', 'Q9Y6Q9',
        ]
    
    def download_structure(self, uniprot_id: str, version: int = 4) -> Optional[str]:
        """
        Download AlphaFold structure for given UniProt ID.
        
        Args:
            uniprot_id: UniProt accession (e.g., 'P04637')
            version: AlphaFold version (default: 4)
            
        Returns:
            Path to downloaded file, or None if failed
        """
        # AlphaFold file naming: AF-{UNIPROT_ID}-F1-model_v{VERSION}.pdb
        filename = f"AF-{uniprot_id}-F1-model_v{version}.pdb"
        output_path = self.output_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"  ✓ Already exists: {filename}")
            return str(output_path)
        
        # Construct download URL
        url = f"{self.BASE_URL}/files/{filename}"
        
        try:
            print(f"  Downloading {filename}...", end=' ')
            urllib.request.urlretrieve(url, output_path)
            print("✓")
            time.sleep(self.delay)  # Be nice to the server
            return str(output_path)
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"✗ (not found)")
            else:
                print(f"✗ (HTTP {e.code})")
            return None
            
        except Exception as e:
            print(f"✗ ({e})")
            return None
    
    def download_batch(self, uniprot_ids: List[str], 
                      max_downloads: Optional[int] = None) -> List[str]:
        """
        Download multiple structures.
        
        Args:
            uniprot_ids: List of UniProt accessions
            max_downloads: Maximum number to download (None = all)
            
        Returns:
            List of successfully downloaded file paths
        """
        downloaded = []
        
        for i, uniprot_id in enumerate(uniprot_ids, 1):
            if max_downloads and len(downloaded) >= max_downloads:
                break
            
            print(f"[{i}/{len(uniprot_ids)}] {uniprot_id}")
            result = self.download_structure(uniprot_id)
            
            if result:
                downloaded.append(result)
        
        return downloaded


def get_random_uniprot_accessions(count: int, seed: Optional[int] = None) -> List[str]:
    """
    Generate random UniProt-like accessions for testing.
    
    Note: In production, you should query UniProt API for real accessions.
    """
    if seed:
        random.seed(seed)
    
    # Common human proteins from AlphaFold DB (real accessions)
    known_accessions = [
        # DNA repair and p53 pathway
        'P04637', 'P38398', 'Q00987', 'P04626',
        # Kinases
        'P24941', 'P31749', 'P31751', 'Q00534',
        # Structural proteins
        'P68431', 'P62258', 'P68363', 'P60709',
        # Enzymes
        'P42574', 'P42575', 'P42576', 'P55210',
        # Receptors
        'P01112', 'P01116', 'P01133', 'P01308',
        # Transcription factors
        'P15336', 'P19838', 'P10276', 'P10275',
        # Additional proteins
        'Q15796', 'Q9Y6Q9', 'O15355', 'Q13485',
        'Q96GQ7', 'O14976', 'Q9UKV8', 'P31946',
        'Q5VSL9', 'Q8N6T3', 'Q9Y6K9', 'P62987',
        'P61981', 'P63104', 'P62993', 'P62857',
        'P62988', 'P61978', 'P46779', 'P62266',
        'P62273', 'P62241', 'P62269', 'P62263',
        'P62277', 'P62280', 'P62851', 'P62854',
        'P62847', 'P62249', 'P62244', 'P62081',
        'P62701', 'P62753', 'P62829', 'P62913',
        'P62906', 'P62910', 'P62945', 'P62979',
        'P63220', 'P63244', 'P63261', 'P61247',
    ]
    
    # If we need more than we have, add some variations
    if count > len(known_accessions):
        print(f"Warning: Only {len(known_accessions)} known accessions available.")
        print(f"Requesting {count}, will return {len(known_accessions)}")
        count = len(known_accessions)
    
    return random.sample(known_accessions, count)


def main():
    parser = argparse.ArgumentParser(
        description="Download random sample of structures from AlphaFold Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 10 random structures
  python testdataset.py --count 10 --output-dir test_structures
  
  # Download 20 structures with a specific seed (reproducible)
  python testdataset.py --count 20 --output-dir data --seed 42
  
  # Download with custom delay between requests
  python testdataset.py --count 5 --output-dir pdbs --delay 1.0
  
  # Download specific proteins (by UniProt accession)
  python testdataset.py --accessions P04637 P01112 P42574 --output-dir structures

Notes:
  - Structures are downloaded from https://alphafold.ebi.ac.uk/
  - Files are named: AF-{UNIPROT_ID}-F1-model_v4.pdb
  - Already downloaded files are skipped
  - A delay between downloads is recommended to be respectful to the server
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--count",
        type=int,
        help="Number of random structures to download"
    )
    input_group.add_argument(
        "--accessions",
        type=str,
        nargs='+',
        help="Specific UniProt accessions to download"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="alphafold_structures",
        help="Output directory for downloaded structures (default: alphafold_structures)"
    )
    
    # Download options
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between downloads in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=4,
        help="AlphaFold model version (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling"
    )
    
    # Filtering options (for future expansion)
    parser.add_argument(
        "--organism",
        type=str,
        default=None,
        choices=list(AlphaFoldDownloader.ORGANISMS.keys()),
        help="Filter by organism (requires querying UniProt - not fully implemented)"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    print(f"AlphaFold Database Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Delay between downloads: {args.delay}s")
    print()
    
    downloader = AlphaFoldDownloader(
        output_dir=args.output_dir,
        delay=args.delay
    )
    
    # Get list of accessions
    if args.accessions:
        accessions = args.accessions
        print(f"Downloading {len(accessions)} specific structures...")
    else:
        print(f"Generating {args.count} random accessions...")
        if args.seed:
            print(f"Using random seed: {args.seed}")
        accessions = get_random_uniprot_accessions(args.count, args.seed)
        print(f"Selected {len(accessions)} accessions")
    
    print()
    
    # Download structures
    downloaded = downloader.download_batch(accessions)
    
    # Summary
    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Successfully downloaded: {len(downloaded)}/{len(accessions)}")
    print(f"Output directory: {args.output_dir}")
    
    if len(downloaded) < len(accessions):
        failed = len(accessions) - len(downloaded)
        print(f"Failed downloads: {failed}")
        print("\nNote: Some proteins may not have AlphaFold predictions available.")
    
    # Write list of downloaded files
    if downloaded:
        list_file = Path(args.output_dir) / "downloaded_files.txt"
        with open(list_file, 'w') as f:
            for path in downloaded:
                f.write(f"{path}\n")
        print(f"\nList of files written to: {list_file}")
    
    print("\n✓ Ready to use with build_trainingset.py!")
    print(f"\nNext steps:")
    print(f"  python -m esm3di.build_trainingset \\")
    print(f"      --pdb-dir {args.output_dir} \\")
    print(f"      --output-prefix training_data \\")
    print(f"      --plddt-threshold 70")


if __name__ == "__main__":
    main()

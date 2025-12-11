#!/usr/bin/env python
"""
Test script to verify PyTorch Lightning setup before submitting to SLURM.
This helps catch configuration errors early.
"""

import sys
import argparse

def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")
    errors = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"✗ PyTorch: {e}")
    
    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning {pl.__version__}")
    except ImportError as e:
        errors.append(f"✗ PyTorch Lightning: {e}")
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        errors.append(f"✗ Transformers: {e}")
    
    try:
        from ESM3di_model import ESM3DiModel, Seq3DiDataset
        print("✓ ESM3di_model")
    except ImportError as e:
        errors.append(f"✗ ESM3di_model: {e}")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  {error}")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    import torch
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        print("  Note: This is OK for CPU-only testing")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available")
    print(f"✓ Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    print()
    return True


def test_config(config_path):
    """Test loading and validating a config file."""
    print(f"Testing config file: {config_path}")
    
    import json
    import os
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✓ Config file loaded successfully")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False
    
    # Check required fields
    required_fields = ['aa_fasta', 'three_di_fasta']
    missing = [f for f in required_fields if f not in config]
    
    if missing:
        print(f"✗ Missing required fields: {missing}")
        return False
    
    # Check if data files exist
    for field in ['aa_fasta', 'three_di_fasta']:
        if field in config:
            path = config[field]
            if os.path.exists(path):
                print(f"✓ {field}: {path}")
            else:
                print(f"⚠ {field} not found: {path}")
    
    # Show key config values
    print("\nKey configuration:")
    important_keys = ['hf_model', 'batch_size', 'epochs', 'lr', 'devices', 'num_nodes']
    for key in important_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    print()
    return True


def test_lightning_script():
    """Test that the Lightning script can be imported."""
    print("Testing Lightning script...")
    
    try:
        import esmretrain_lightning
        print("✓ esmretrain_lightning.py can be imported")
        print()
        return True
    except Exception as e:
        print(f"✗ Error importing esmretrain_lightning.py: {e}")
        print()
        return False


def test_data_loading(config_path):
    """Test data loading with a small sample."""
    print("Testing data loading...")
    
    import json
    from ESM3di_model import Seq3DiDataset
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        aa_fasta = config.get('aa_fasta', 'test_data_aa.fasta')
        three_di_fasta = config.get('three_di_fasta', 'test_data_3di.fasta')
        mask_label_chars = config.get('mask_label_chars', 'X')
        
        dataset = Seq3DiDataset(aa_fasta, three_di_fasta, mask_label_chars=mask_label_chars)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of sequences: {len(dataset)}")
        print(f"  3Di vocabulary ({len(dataset.label_vocab)}): {dataset.label_vocab}")
        
        if len(dataset) > 0:
            header, seq, di = dataset[0]
            print(f"\n  Sample sequence:")
            print(f"    Header: {header[:50]}...")
            print(f"    AA length: {len(seq)}")
            print(f"    3Di length: {len(di)}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Lightning setup before SLURM submission")
    parser.add_argument("--config", type=str, default="config_lightning.json",
                       help="Path to config file to test")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data loading test (useful if data files are on compute nodes only)")
    args = parser.parse_args()
    
    print("="*60)
    print("ESM3di Lightning Setup Test")
    print("="*60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Config", test_config(args.config)))
    results.append(("Lightning Script", test_lightning_script()))
    
    if not args.skip_data:
        results.append(("Data Loading", test_data_loading(args.config)))
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("✓ All tests passed! Ready to submit to SLURM.")
        print()
        print("Next steps:")
        print("  1. Edit slurm_train.sh with your cluster settings")
        print("  2. sbatch slurm_train.sh")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before submitting.")
        print()
        print("Common fixes:")
        print("  - Install missing packages: pip install pytorch-lightning")
        print("  - Check config file paths")
        print("  - Verify data files exist")
        return 1


if __name__ == "__main__":
    sys.exit(main())

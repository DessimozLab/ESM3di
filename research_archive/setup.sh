#!/bin/bash
# Quick setup script for ESM3Di

echo "Setting up ESM3Di environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found"

# Create conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment and start using ESM3Di:"
echo "  conda activate esm3di"
echo ""
echo "To train a model:"
echo "  python -m esm3di.esmretrain --aa-fasta data/sequences.fasta --three-di-fasta data/3di_labels.fasta"
echo ""
echo "For more information, see README.md"

#!/bin/bash --login
#SBATCH --job-name=3di_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --partition=gpu          # Official UNIL DCSR GPU partition name
#SBATCH --gres=gpu:1             # Requests 1 full A100 GPU (or use: --gpus=1)
#SBATCH --cpus-per-task=8        # Allocates 8 CPU cores for data processing
#SBATCH --mem=32G                # Allocates 32 GB RAM
#SBATCH --time=04:00:00          # 4 hours wall-time limit

# Activate Conda environment cleanly
conda activate esm3di

# Use fast local node scratch space for temporary FASTAs/outputs
export TMPDIR="/tmp"

# Execute benchmark
python run_benchmark.py --models esm3di --repeats 5 --results-file results/new_nseq_esm3di_cli.csv --subsets 10 30 100 300 1000 3000 10000

#python run_benchmark.py --models esm3di prostt5 --length-bins --repeats 6 --results-file results/final_length_bins_plms.csv --upsample-nseqs-to 512

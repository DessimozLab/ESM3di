#!/bin/bash --login
#SBATCH --job-name=3di_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --partition=gpu          # Official UNIL DCSR GPU partition name
#SBATCH --gres=gpu:1             # Requests 1 full A100 GPU (or use: --gpus=1)
#SBATCH --cpus-per-task=8        # Allocates 8 CPU cores for data processing
#SBATCH --mem=32G                # Allocates 32 GB RAM
#SBATCH --time=05:00:00          # 4 hours wall-time limit

# 1. Activate your primary benchmark Python environment
conda activate esm3di

# 2. Redirect package manager caches to scratch space (prevents home quota errors)
export PIXI_CACHE_DIR="/scratch/$USER/.cache/pixi"
export UV_CACHE_DIR="/scratch/$USER/.cache/uv"
export RATTLER_CACHE_DIR="/scratch/$USER/.cache/rattler"

# 3. Add LocalColabFold binaries to PATH
export PATH="$PATH:/scratch/$USER/localcolabfold/localcolabfold/.pixi/envs/default/bin"

# 4. Point ColabFold to the model parameters on scratch
export COLABFOLD_DIR="/scratch/$USER/localcolabfold/colabfold_weights"

# 5. Ensure temporary files use node local storage
export TMPDIR="/tmp"

# Execute benchmark
python run_benchmark.py --models colabfold --subsets 10 30 100 300 --repeats 2 --results-file results/final_nseq.csv
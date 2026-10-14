#!/bin/bash --login
#SBATCH --job-name=3di_metagenomic
#SBATCH --output=log_esm3di_%j.out
#SBATCH --error=log_esm3di_%j.err
#SBATCH --partition=gpu          # Official UNIL DCSR GPU partition name
#SBATCH --gres=gpu:1             # Requests 1 full A100 GPU (or use: --gpus=1)
#SBATCH --cpus-per-task=8        # Allocates 8 CPU cores for data processing
#SBATCH --mem=32G                # Allocates 32 GB RAM
#SBATCH --time=04:00:00          # 4 hours wall-time limit

# Activate Conda environment cleanly
conda activate esm3di

# Use fast local node scratch space for temporary FASTAs/outputs
export TMPDIR="/tmp"

esm3di foldseek-db \
    --input-fasta /work/FAC/FBM/DBC/cdessim2/default/dkim5/projects/viral/metavr/clu/IMGVR_PC_000476333.faa \
    --output-db /work/FAC/FBM/DBC/cdessim2/default/fweikert/IMGVR_PC_000476333/IMGVR_PC_000476333 \
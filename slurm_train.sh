#!/bin/bash
#SBATCH --job-name=esm3di_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu

# SLURM Single-Node Multi-GPU Training Script for ESM3di
# This script trains on multiple GPUs on a single node

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment
# source venv/bin/activate
# OR if using conda:
# conda activate esm3di

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Prevent NCCL timeout errors
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Configuration file
CONFIG_FILE="config_example.json"

# Number of GPUs (detected automatically from SLURM)
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Number of GPUs: $NUM_GPUS"
echo ""

# Run training with PyTorch Lightning using torchrun
echo "Starting training..."
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    esmretrain_lightning.py \
    --config "$CONFIG_FILE" \
    --devices $NUM_GPUS \
    --num-nodes 1 \
    --strategy ddp \
    --precision bf16-mixed \
    --num-workers $SLURM_CPUS_PER_TASK

echo ""
echo "Training completed!"
echo "Check logs at: logs/slurm-$SLURM_JOB_ID.out"

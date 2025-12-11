#!/bin/bash
#SBATCH --job-name=esm3di_multinode
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu

# SLURM Multi-Node Multi-GPU Training Script for ESM3di
# This script trains across multiple nodes with multiple GPUs per node

echo "=========================================="
echo "SLURM Multi-Node Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10
# module load nccl/2.15

# Activate virtual environment
# source venv/bin/activate
# OR if using conda:
# conda activate esm3di

# Set environment variables for optimal multi-node performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL settings for multi-node training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch Lightning will handle the distributed setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12910

# Configuration file
CONFIG_FILE="config_example.json"

# Number of nodes and GPUs
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Master node: $MASTER_ADDR:$MASTER_PORT"
echo "  Number of nodes: $NUM_NODES"
echo "  GPUs per node: $NUM_GPUS"
echo "  Total GPUs: $((NUM_NODES * NUM_GPUS))"
echo ""

# Run training with torchrun for multi-node DDP
echo "Starting multi-node training..."
echo ""

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    esmretrain_lightning.py \
    --config "$CONFIG_FILE" \
    --devices $NUM_GPUS \
    --num-nodes $NUM_NODES \
    --strategy ddp \
    --precision bf16-mixed \
    --num-workers $SLURM_CPUS_PER_TASK

echo ""
echo "Training completed!"
echo "Check logs at: logs/slurm-$SLURM_JOB_ID.out"

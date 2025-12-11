#!/bin/bash
#SBATCH --job-name=esm3di_fsdp
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu

# SLURM FSDP (Fully Sharded Data Parallel) Training Script
# This script uses FSDP for training very large models that don't fit in GPU memory
# FSDP shards model parameters, gradients, and optimizer states across GPUs

echo "=========================================="
echo "SLURM FSDP Training Job Information"
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

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_ASYNC_ERROR_HANDLING=1

# Master node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12910

# Configuration file
CONFIG_FILE="config_example.json"

# Number of nodes and GPUs
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "FSDP Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Master node: $MASTER_ADDR:$MASTER_PORT"
echo "  Number of nodes: $NUM_NODES"
echo "  GPUs per node: $NUM_GPUS"
echo "  Total GPUs: $((NUM_NODES * NUM_GPUS))"
echo ""

# Run training with torchrun for FSDP
echo "Starting FSDP training..."
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
    --strategy fsdp \
    --precision bf16-mixed \
    --num-workers $SLURM_CPUS_PER_TASK

echo ""
echo "Training completed!"
echo "Check logs at: logs/slurm-$SLURM_JOB_ID.out"

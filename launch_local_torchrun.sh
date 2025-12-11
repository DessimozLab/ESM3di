#!/bin/bash

# Local Torchrun Training Script for ESM3di
# This script allows you to test multi-GPU training locally before submitting to SLURM

set -e  # Exit on error

# Default values
NUM_GPUS=2
CONFIG_FILE=""

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --num-gpus NUM        Number of GPUs to use (default: 2)"
    echo "  -c, --config FILE         Path to config JSON file (required)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Train on 2 GPUs using config.json"
    echo "  $0 -n 2 -c config_lightning.json"
    echo ""
    echo "  # Train on all available GPUs"
    echo "  $0 -n -1 -c config_lightning.json"
    echo ""
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file is required (-c/--config)"
    usage
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if CUDA is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Error: CUDA is not available. Multi-GPU training requires CUDA."
    exit 1
fi

# Get number of available GPUs
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Available GPUs: $AVAILABLE_GPUS"

# Handle -1 (use all GPUs)
if [ "$NUM_GPUS" -eq -1 ]; then
    NUM_GPUS=$AVAILABLE_GPUS
    echo "Using all available GPUs: $NUM_GPUS"
fi

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "         Using $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: Must use at least 1 GPU"
    exit 1
fi

# Print configuration
echo ""
echo "========================================"
echo "Local Torchrun Training Configuration"
echo "========================================"
echo "Config file:     $CONFIG_FILE"
echo "Number of GPUs:  $NUM_GPUS"
echo "========================================"
echo ""

# Launch training with torchrun
echo "Launching training with torchrun..."
echo ""

# For single-node training, use --standalone flag
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU - no need for torchrun
    python esmretrain_lightning.py \
        --config "$CONFIG_FILE" \
        --devices 1 \
        --num-nodes 1
else
    # Multi-GPU - use torchrun with standalone mode
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        esmretrain_lightning.py \
        --config "$CONFIG_FILE" \
        --devices $NUM_GPUS \
        --num-nodes 1 \
        --strategy ddp
fi

echo ""
echo "Training completed!"

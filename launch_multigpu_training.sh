#!/bin/bash

# Multi-GPU Training Launch Script for ESM3di
# This script makes it easy to launch distributed training across multiple GPUs

set -e  # Exit on error

# Default values
NUM_GPUS=2
CONFIG_FILE=""
GPU_IDS=""

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --num-gpus NUM        Number of GPUs to use (default: 2)"
    echo "  -c, --config FILE         Path to config JSON file (required)"
    echo "  -g, --gpu-ids IDS         Comma-separated GPU IDs to use (e.g., '0,1,2')"
    echo "                            If not specified, uses first NUM_GPUS available"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Train on 2 GPUs using config.json"
    echo "  $0 -n 2 -c config.json"
    echo ""
    echo "  # Train on 4 GPUs"
    echo "  $0 -n 4 -c config_example.json"
    echo ""
    echo "  # Train on specific GPUs (0 and 2)"
    echo "  $0 -n 2 -c config.json -g 0,2"
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
        -g|--gpu-ids)
            GPU_IDS="$2"
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

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "         Using $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Set CUDA_VISIBLE_DEVICES if specific GPUs requested
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using GPUs: $GPU_IDS"
    # Count the number of GPUs specified
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
else
    echo "Using first $NUM_GPUS GPUs"
fi

# Print configuration
echo ""
echo "========================================"
echo "Multi-GPU Training Configuration"
echo "========================================"
echo "Config file:     $CONFIG_FILE"
echo "Number of GPUs:  $NUM_GPUS"
if [ -n "$GPU_IDS" ]; then
    echo "GPU IDs:         $GPU_IDS"
fi
echo "========================================"
echo ""

# Launch training with torchrun
echo "Launching training..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    esmretrain_multigpu.py \
    --config "$CONFIG_FILE"

echo ""
echo "Training completed!"

#!/bin/bash
# Train ESM++ small clade-specific models sequentially
# Usage: ./train_all_classes.sh [--resume]
#
# Each model will be trained using the corresponding config file.
# Use --resume to continue from the last checkpoint if available.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configs in order of dataset size (largest first)
CONFIG_ENTRIES=(
    "caudoviricetes:config_esmpp_small_caudoviricetes.json:checkpoints_esmpp_small_caudoviricetes"
    "megaviricetes:config_esmpp_small_megaviricetes.json:checkpoints_esmpp_small_megaviricetes"
    "herviviricetes:config_esmpp_small_herviviricetes.json:checkpoints_esmpp_small_herviviricetes"
    "naldaviricetes:config_esmpp_small_naldaviricetes.json:checkpoints_esmpp_small_naldaviricetes"
    "pokkesviricetes:config_esmpp_small_pokkesviricetes.json:checkpoints_esmpp_small_pokkesviricetes"
    "pisoniviricetes:config_esmpp_small_pisoniviricetes.json:checkpoints_esmpp_small_pisoniviricetes"
    "malgrandaviricetes:config_esmpp_small_malgrandaviricetes.json:checkpoints_esmpp_small_malgrandaviricetes"
    "revtraviricetes:config_esmpp_small_revtraviricetes.json:checkpoints_esmpp_small_revtraviricetes"
    "alsuviricetes:config_esmpp_small_alsuviricetes.json:checkpoints_esmpp_small_alsuviricetes"
    "monjiviricetes:config_esmpp_small_monjiviricetes.json:checkpoints_esmpp_small_monjiviricetes"
    "flavi:config_esmpp_small_full_transformer_flavi.json:checkpoints_esmpp_small_flavi"
)

# Parse arguments
RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume-from-checkpoint"
    echo "Resume mode enabled - will resume from latest checkpoint if available"
fi

echo "========================================"
echo "ESM3Di Class-specific Model Training"
echo "========================================"
echo "Training ${#CONFIG_ENTRIES[@]} models sequentially"
echo "Start time: $(date)"
echo ""

# Track progress
COMPLETED=0
FAILED=0
SKIPPED=0

for ENTRY in "${CONFIG_ENTRIES[@]}"; do
    IFS=':' read -r CLASS CONFIG_FILE CHECKPOINT_DIR <<< "$ENTRY"
    
    echo "========================================"
    echo "Training: $CLASS"
    echo "Config: $CONFIG_FILE"
    echo "Started: $(date)"
    echo "========================================"
    
    # Check if config exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        ((FAILED++))
        continue
    fi
    
    # Find latest checkpoint for resume
    RESUME_CKPT=""
    if [[ -n "$RESUME_FLAG" && -d "$CHECKPOINT_DIR" ]]; then
        # Find the highest epoch checkpoint
        LATEST=$(ls -v "$CHECKPOINT_DIR"/epoch_*.pt 2>/dev/null | grep -v "_model.pt" | tail -1)
        if [[ -n "$LATEST" ]]; then
            RESUME_CKPT="$RESUME_FLAG $LATEST"
            echo "Resuming from: $LATEST"
        fi
    fi
    
    # Run training
    if python -m esm3di.esmretrain --config "$CONFIG_FILE" $RESUME_CKPT; then
        echo "✓ Completed: $CLASS"
        ((COMPLETED++))
    else
        echo "✗ Failed: $CLASS"
        ((FAILED++))
    fi
    
    echo ""
done

echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "End time: $(date)"
echo "Completed: $COMPLETED / ${#CONFIG_ENTRIES[@]}"
echo "Failed: $FAILED"
echo ""
echo "Checkpoints saved to:"
for ENTRY in "${CONFIG_ENTRIES[@]}"; do
    IFS=':' read -r CLASS CONFIG_FILE CHECKPOINT_DIR <<< "$ENTRY"
    echo "  - ${CHECKPOINT_DIR}/"
done

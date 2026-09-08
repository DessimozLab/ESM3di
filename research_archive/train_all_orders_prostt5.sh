#!/bin/bash
# Train ProstT5 models on all 10 order-level datasets
# Usage: ./train_all_orders_prostt5.sh [--resume]

set -e

ORDERS=(
    "imitervirales"
    "algavirales"
    "herpesvirales"
    "pimascovirales"
    "lefavirales"
    "chitovirales"
    "petitvirales"
    "ortervirales"
    "picornavirales"
    "mononegavirales"
)

RESUME_MODE=false
if [[ "$1" == "--resume" ]]; then
    RESUME_MODE=true
    echo "Resume mode enabled - will continue from last checkpoint"
fi

# Function to find the latest checkpoint in a directory
find_latest_checkpoint() {
    local ckpt_dir="$1"
    if [[ -d "$ckpt_dir" ]]; then
        # Find the highest epoch checkpoint (epoch_N.pt format)
        local latest=$(ls -1 "${ckpt_dir}"/epoch_*.pt 2>/dev/null | \
            grep -E 'epoch_[0-9]+\.pt$' | \
            sort -t_ -k2 -n | \
            tail -1)
        echo "$latest"
    fi
}

echo "=============================================="
echo "Training ProstT5 on 10 viral orders"
echo "=============================================="
echo ""

for order in "${ORDERS[@]}"; do
    echo "=============================================="
    echo "Training: $order"
    echo "Config: config_prostt5_${order}.json"
    echo "=============================================="
    
    RESUME_ARG=""
    if [[ "$RESUME_MODE" == true ]]; then
        ckpt_dir="checkpoints_prostt5_${order}"
        latest_ckpt=$(find_latest_checkpoint "$ckpt_dir")
        if [[ -n "$latest_ckpt" && -f "$latest_ckpt" ]]; then
            echo "Resuming from: $latest_ckpt"
            RESUME_ARG="--resume-from-checkpoint $latest_ckpt"
        else
            echo "No checkpoint found in $ckpt_dir - starting fresh"
        fi
    fi
    
    python -m esm3di.esmretrain \
        --config "config_prostt5_${order}.json" \
        $RESUME_ARG
    
    echo ""
    echo "Completed: $order"
    echo ""
done

echo "=============================================="
echo "All training complete!"
echo "=============================================="
echo ""
echo "Checkpoints saved to:"
for order in "${ORDERS[@]}"; do
    echo "  - checkpoints_prostt5_${order}/"
done

from safetensors.torch import load_file
import json

# Replace with your actual hf_compatible folder path
safetensor_path = "esm3di_inference/checkpoints/hf_compatible/adapter_model.safetensors"

import torch
from pathlib import Path

# Update these paths to your actual local paths
ORIGINAL_CHECKPOINT_PATH = "esm3di_inference/checkpoints/epoch_5.pt"
HF_COMPATIBLE_FOLDER_PATH = "esm3di_inference/checkpoints/hf_compatible"

# 1. Load the unified training checkpoint
checkpoint = torch.load(ORIGINAL_CHECKPOINT_PATH, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)

# 2. Extract only the custom cnn_head weights
cnn_head_weights = {}
for k, v in state_dict.items():
    # Remove DataParallel prefixes if present
    k_clean = k.replace("module.", "", 1)
    if k_clean.startswith("cnn_head."):
        # Strip the prefix "cnn_head." so they load directly into the Module
        module_key = k_clean.replace("cnn_head.", "", 1)
        cnn_head_weights[module_key] = v

# 3. Save the clean weights inside your hf_compatible folder
output_path = Path(HF_COMPATIBLE_FOLDER_PATH) / "cnn_head.bin"
torch.save(cnn_head_weights, output_path)
print(f"✓ Extracted {len(cnn_head_weights)} CNN layers and saved to {output_path}")
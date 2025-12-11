# Uploading ESM3Di Models to Hugging Face Hub

This guide explains how to upload your trained ESM3Di models to the Hugging Face Hub for easy sharing and deployment.

**Important**: The upload script automatically merges LoRA weights into the base model before uploading, creating a standalone model that doesn't require the PEFT library for inference.

## Prerequisites

1. Install the Hugging Face Hub library:
```bash
pip install huggingface-hub
```

2. Get your Hugging Face API token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "write" access
   - Copy the token

3. Login (optional, can also pass token directly):
```bash
huggingface-cli login
```

## Basic Usage

Upload a trained model checkpoint:

```bash
python upload_to_hf.py \
    --checkpoint checkpoints/epoch_10.pt \
    --repo-id your-username/esm3di-model
```

## Full Example with Metadata

```bash
python upload_to_hf.py \
    --checkpoint checkpoints/epoch_10.pt \
    --repo-id your-username/esm3di-650m-foldseek \
    --model-name "ESM3Di 650M Foldseek Trained" \
    --author "Your Name" \
    --contact "your.email@example.com" \
    --dataset-description "Trained on X protein structures from PDB/AlphaFoldDB" \
    --private  # Make repository private (remove for public)
```

## Arguments

### Required
- `--checkpoint`: Path to your trained model checkpoint (`.pt` file)
- `--repo-id`: Hugging Face repository ID (format: `username/model-name`)

### Optional
- `--model-name`: Human-readable name for the model card
- `--token`: HF API token (if not logged in via `huggingface-cli login`)
- `--private`: Make the repository private
- `--output-dir`: Temporary directory for preparing files (default: `hf_upload_temp`)
- `--author`: Your name for the model card
- `--contact`: Contact information
- `--dataset-description`: Description of your training data

## What Gets Uploaded

The script automatically:
1. **Checks if LoRA weights are merged** - If not, merges them automatically
2. **Uploads merged model** - No PEFT/LoRA dependencies needed for inference
3. **Creates comprehensive documentation**:
   - **pytorch_model.bin** - Merged model weights
   - **checkpoint.pt** - Full merged checkpoint with metadata
   - **config.json** - Model configuration (marked as merged)
   - **README.md** - Auto-generated model card
   - **training_info.json** - Training metrics

**Benefits of merged models**:
- ✅ Smaller file size (no LoRA metadata)
- ✅ Faster loading (no adapter merging at runtime)
- ✅ No PEFT library required for inference
- ✅ Standard transformers model architecture

## Using Uploaded Models

### Download and Load (No PEFT needed!)

```python
from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification
import torch

# Download merged checkpoint
checkpoint_path = hf_hub_download(
    repo_id="your-username/esm3di-model",
    filename="checkpoint.pt"
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
config = checkpoint['config']

# Initialize standard transformers model
model = AutoModelForTokenClassification.from_pretrained(
    config['base_model'],
    num_labels=config['num_labels']
)

# Load merged weights (no PEFT library needed!)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
# ... your inference code ...
```

### Resume Training from Hub

Note: Downloaded models have merged weights. To resume training, you'll need to 
re-apply LoRA adapters or use the original checkpoint.

```python
from huggingface_hub import hf_hub_download

# This won't work for training continuation (weights are merged)
checkpoint = hf_hub_download(
    repo_id="your-username/esm3di-model",
    filename="checkpoint.pt"
)

# For training, keep your original LoRA checkpoints locally!
```

## Environment Variables

Instead of passing `--token`, you can set:
```bash
export HF_TOKEN="your_token_here"
```

## Tips

1. **Choose meaningful repo names**: Use descriptive names like `esm3di-650m-pdb` or `esm3di-3b-alphafold`

2. **Document your training data**: The more details in `--dataset-description`, the more useful for others

3. **Use private repos for experiments**: You can always make them public later

4. **Version your models**: Use different repo IDs or branches for different training runs

## Troubleshooting

**Authentication Error**: Make sure you're logged in or passing a valid token
```bash
huggingface-cli login
```

**Upload Timeout**: For large models, increase timeout or check your internet connection

**Repository Already Exists**: The script will update existing repos (use `--private` carefully!)

## Example Workflow

```bash
# 1. Train your model
python esmretrain.py --config config.json

# 2. Upload best checkpoint
python upload_to_hf.py \
    --checkpoint checkpoints/epoch_10.pt \
    --repo-id your-username/my-esm3di-model \
    --model-name "ESM3Di Custom Dataset" \
    --dataset-description "Trained on 50K protein structures"

# 3. Share the link!
# https://huggingface.co/your-username/my-esm3di-model
```

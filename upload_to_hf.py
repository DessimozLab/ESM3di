#!/usr/bin/env python3
"""
Script to upload trained ESM3Di models to Hugging Face Hub.

This script handles:
- Model card generation with training details
- Uploading model weights and configuration
- Setting up proper model repository structure
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

from ESM3di_model import merge_lora_weights


def create_model_card(args, checkpoint_info):
    """
    Generate a comprehensive model card for the uploaded model.
    """
    model_card = f"""---
language: en
license: mit
tags:
- protein
- 3di
- structure
- esm
- lora
datasets:
- custom
metrics:
- accuracy
library_name: transformers
pipeline_tag: token-classification
---

# ESM3Di: {args.model_name}

This model predicts 3Di (3D interaction) sequences from amino acid sequences.
It's a fine-tuned version of `{checkpoint_info.get('base_model', 'ESM-2')}` using LoRA adaptation.

**Note**: This model has been uploaded with LoRA weights merged into the base model, 
so it can be used without the PEFT library.

## Model Description

- **Base Model**: {checkpoint_info.get('base_model', 'facebook/esm2_t33_650M_UR50D')}
- **Task**: Token classification (per-residue 3Di prediction)
- **Training Method**: LoRA (Low-Rank Adaptation) - weights now merged
- **Number of Labels**: {checkpoint_info.get('num_labels', 20)}
- **Original LoRA Rank**: {checkpoint_info.get('lora_r', 8)}
- **Original LoRA Alpha**: {checkpoint_info.get('lora_alpha', 16)}

## Training Details

### Training Data
{args.dataset_description if args.dataset_description else "Custom protein dataset with amino acid and 3Di sequence pairs."}

### Training Hyperparameters

- **Epochs**: {checkpoint_info.get('epoch', 'N/A')}
- **Batch Size**: {checkpoint_info.get('batch_size', 'N/A')}
- **Learning Rate**: {checkpoint_info.get('learning_rate', 'N/A')}
- **Weight Decay**: {checkpoint_info.get('weight_decay', 'N/A')}
- **Scheduler**: {checkpoint_info.get('scheduler_type', 'N/A')}
- **Final Loss**: {checkpoint_info.get('loss', 'N/A'):.4f if isinstance(checkpoint_info.get('loss'), (int, float)) else 'N/A'}

### Model Architecture

- **CNN Head**: {checkpoint_info.get('use_cnn_head', False)}
- **LoRA Target Modules**: {', '.join(checkpoint_info.get('lora_target_modules', [])) if checkpoint_info.get('lora_target_modules') else 'auto-discovered'}
- **Masked Label Characters**: {checkpoint_info.get('mask_label_chars', 'none')}

## Usage

### Option 1: Using ESM3Di wrapper (recommended)

```python
from huggingface_hub import hf_hub_download
import torch

# Download merged checkpoint
checkpoint_path = hf_hub_download(
    repo_id="{args.repo_id}",
    filename="checkpoint.pt"
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
config = checkpoint['config']

# Initialize base model (no LoRA needed - weights are merged!)
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    config['base_model'],
    num_labels=config['num_labels']
)

# Load merged weights
model.load_state_dict(checkpoint['model_state_dict'])

# Use for prediction
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
# ... your inference code ...
```

### Option 2: Using ESM3DiModel wrapper

```python
from ESM3di_model import ESM3DiModel
from huggingface_hub import hf_hub_download
import torch

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="{args.repo_id}",
    filename="checkpoint.pt"
)

# Initialize model (config will be loaded from checkpoint)
checkpoint = torch.load(checkpoint_path)
config = checkpoint['config']

model = AutoModelForTokenClassification.from_pretrained(
    config['base_model'],
    num_labels=config['num_labels']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Use model for inference
# Note: No PEFT/LoRA library needed!
```

## Citation

If you use this model, please cite:

```bibtex
@software{{esm3di,
  title = {{ESM3Di: 3Di Structure Sequence Prediction}},
  author = {{{args.author if args.author else 'Your Name'}}},
  year = {{2025}},
  url = {{https://huggingface.co/{args.repo_id}}}
}}
```

## Model Card Contact

For questions or issues, please contact {args.contact if args.contact else 'the repository maintainer'}.
"""
    return model_card


def extract_checkpoint_info(checkpoint_path):
    """
    Extract relevant information from checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {}
    
    # Direct checkpoint fields
    info['epoch'] = checkpoint.get('epoch', 'N/A')
    info['loss'] = checkpoint.get('loss', 'N/A')
    info['label_vocab'] = checkpoint.get('label_vocab', [])
    info['num_labels'] = len(checkpoint.get('label_vocab', []))
    info['mask_label_chars'] = checkpoint.get('mask_label_chars', '')
    info['lora_target_modules'] = checkpoint.get('lora_target_modules', [])
    
    # From args saved in checkpoint
    args_dict = checkpoint.get('args', {})
    info['base_model'] = args_dict.get('hf_model_name', 'facebook/esm2_t33_650M_UR50D')
    info['batch_size'] = args_dict.get('batch_size', 'N/A')
    info['learning_rate'] = args_dict.get('lr', 'N/A')
    info['weight_decay'] = args_dict.get('weight_decay', 'N/A')
    info['scheduler_type'] = args_dict.get('scheduler_type', 'N/A')
    info['lora_r'] = args_dict.get('lora_r', 8)
    info['lora_alpha'] = args_dict.get('lora_alpha', 16)
    info['use_cnn_head'] = args_dict.get('use_cnn_head', False)
    
    return info


def prepare_upload_directory(checkpoint_path, output_dir, checkpoint_info, args):
    """
    Prepare a directory with all files to upload.
    Merges LoRA weights into base model before uploading.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if checkpoint is already merged
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    is_merged = checkpoint.get('merged', False)
    
    if is_merged:
        print(f"✓ Checkpoint is already merged")
        merged_checkpoint_path = checkpoint_path
    else:
        print(f"⚠ Checkpoint contains LoRA weights (not merged)")
        print(f"Merging LoRA weights into base model...")
        
        # Merge LoRA weights
        merged_checkpoint_path = os.path.join(output_dir, 'merged_temp.pt')
        merge_lora_weights(
            checkpoint_path=checkpoint_path,
            output_path=merged_checkpoint_path,
            device='cpu'
        )
        
        # Reload merged checkpoint
        checkpoint = torch.load(merged_checkpoint_path, map_location='cpu')
        print(f"✓ LoRA weights merged successfully")
    
    # Save model weights as pytorch_model.bin
    model_state = checkpoint['model_state_dict']
    torch.save(model_state, os.path.join(output_dir, 'pytorch_model.bin'))
    print(f"✓ Saved model weights to {output_dir}/pytorch_model.bin")
    
    # Save merged checkpoint
    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.pt'))
    print(f"✓ Saved merged checkpoint to {output_dir}/checkpoint.pt")
    
    # Clean up temporary merged file if created
    if not is_merged and os.path.exists(merged_checkpoint_path):
        os.remove(merged_checkpoint_path)
    
    # Update config to reflect that this is a merged model
    config = {
        "model_type": "esm3di",
        "base_model": checkpoint_info.get('base_model'),
        "num_labels": checkpoint_info.get('num_labels'),
        "label_vocab": checkpoint_info.get('label_vocab'),
        "use_cnn_head": checkpoint_info.get('use_cnn_head'),
        "mask_label_chars": checkpoint_info.get('mask_label_chars'),
        "merged": True,
        "original_lora_config": {
            "lora_r": checkpoint_info.get('lora_r'),
            "lora_alpha": checkpoint_info.get('lora_alpha'),
            "lora_target_modules": checkpoint_info.get('lora_target_modules'),
        },
        "training_args": checkpoint.get('args', {}),
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config to {output_dir}/config.json")
    
    # Create model card
    model_card = create_model_card(args, checkpoint_info)
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(model_card)
    print(f"✓ Saved model card to {output_dir}/README.md")
    
    # Create training_info.json with additional details
    training_info = {
        "epoch": checkpoint_info.get('epoch'),
        "final_loss": float(checkpoint_info.get('loss')) if isinstance(checkpoint_info.get('loss'), (int, float)) else None,
        "global_step": checkpoint.get('global_step', 'N/A'),
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✓ Saved training info to {output_dir}/training_info.json")


def upload_to_hub(repo_id, upload_dir, token, private=True):
    """
    Upload prepared directory to Hugging Face Hub.
    """
    print(f"\n{'='*60}")
    print(f"Uploading to Hugging Face Hub: {repo_id}")
    print(f"{'='*60}\n")
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        print(f"Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository created/verified")
    except Exception as e:
        print(f"Warning: Could not create repository: {e}")
        print("Attempting to upload anyway...")
    
    # Upload entire folder
    try:
        print(f"\nUploading files from {upload_dir}...")
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"\n✓ Successfully uploaded model to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload trained ESM3Di model to Hugging Face Hub"
    )
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face repository ID (e.g., 'username/model-name')")
    
    # Optional arguments
    parser.add_argument("--model-name", type=str, default=None,
                        help="Human-readable model name for model card")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token. If not provided, uses HF_TOKEN env var or cached credentials")
    parser.add_argument("--private", action="store_true",
                        help="Make repository private")
    parser.add_argument("--output-dir", type=str, default="hf_upload_temp",
                        help="Temporary directory for preparing files")
    
    # Model card metadata
    parser.add_argument("--author", type=str, default=None,
                        help="Author name for model card")
    parser.add_argument("--contact", type=str, default=None,
                        help="Contact information for model card")
    parser.add_argument("--dataset-description", type=str, default=None,
                        help="Description of training dataset for model card")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Set model name from repo_id if not provided
    if args.model_name is None:
        args.model_name = args.repo_id.split('/')[-1]
    
    # Get token
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("Warning: No Hugging Face token provided.")
        print("Using cached credentials if available.")
        print("To set a token, use --token or set HF_TOKEN environment variable.")
    
    print(f"{'='*60}")
    print(f"ESM3Di Model Upload to Hugging Face Hub")
    print(f"{'='*60}\n")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Repository: {args.repo_id}")
    print(f"Private: {args.private}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nNote: The upload script will automatically merge LoRA weights")
    print(f"into the base model if they haven't been merged yet.\n")
    
    # Extract checkpoint information
    print("Extracting checkpoint information...")
    checkpoint_info = extract_checkpoint_info(args.checkpoint)
    print(f"✓ Checkpoint info extracted")
    print(f"  Base model: {checkpoint_info.get('base_model')}")
    print(f"  Epoch: {checkpoint_info.get('epoch')}")
    print(f"  Loss: {checkpoint_info.get('loss')}")
    print(f"  Num labels: {checkpoint_info.get('num_labels')}\n")
    
    # Prepare upload directory
    print("Preparing files for upload...")
    prepare_upload_directory(args.checkpoint, args.output_dir, checkpoint_info, args)
    print()
    
    # Upload to hub
    success = upload_to_hub(args.repo_id, args.output_dir, token, args.private)
    
    if success:
        print(f"\n{'='*60}")
        print("Upload Complete!")
        print(f"{'='*60}")
        print(f"\nYour model is available at:")
        print(f"https://huggingface.co/{args.repo_id}")
        print(f"\nTo use it:")
        print(f"```python")
        print(f"from huggingface_hub import hf_hub_download")
        print(f"checkpoint_path = hf_hub_download(repo_id='{args.repo_id}', filename='checkpoint.pt')")
        print(f"```")
    else:
        print("\nUpload failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

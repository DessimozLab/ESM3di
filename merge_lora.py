#!/usr/bin/env python3
"""
Example script for merging LoRA weights into base model.

This creates a standalone model without LoRA adapters that can be
used for inference without the PEFT library.
"""

from ESM3di_model import merge_lora_weights


def main():
    # Path to your LoRA checkpoint
    checkpoint_path = "checkpoints/epoch_10.pt"
    
    # Path to save merged model
    output_path = "merged_models/esm3di_merged.pt"
    
    print("Merging LoRA weights into base model...")
    print("=" * 60)
    
    # Merge LoRA weights
    merged_model = merge_lora_weights(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        device="cuda"  # or "cpu"
    )
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("\nThe merged model:")
    print("- Has LoRA weights integrated into base model weights")
    print("- Does NOT require PEFT library for inference")
    print("- Can be used like a standard transformers model")
    print("- Is typically smaller than LoRA checkpoint")
    
    print("\nTo use the merged model:")
    print("```python")
    print("import torch")
    print("from transformers import AutoModelForTokenClassification")
    print()
    print("# Load merged checkpoint")
    print(f"checkpoint = torch.load('{output_path}')")
    print("config = checkpoint['config']")
    print()
    print("# Initialize base model")
    print("model = AutoModelForTokenClassification.from_pretrained(")
    print("    config['hf_model_name'],")
    print("    num_labels=config['num_labels']")
    print(")")
    print()
    print("# Load merged weights")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    print("```")


if __name__ == "__main__":
    main()

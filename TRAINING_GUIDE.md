# ESM 3Di Training Guide

This guide explains how to use `esmretrain.py` for training ESM models on 3Di structure prediction.

## Quick Start

### Basic Usage

```bash
python esmretrain.py \
  --aa-fasta test_data_aa.fasta \
  --three-di-fasta test_data_3di_masked.fasta \
  --epochs 3 \
  --batch-size 2
```

### Using a Config File

Create a JSON config file (see `config_example.json`):

```bash
python esmretrain.py --config config_example.json
```

## Key Features

### 1. Dynamic LoRA Target Module Discovery

The script now automatically discovers which modules to apply LoRA to by scanning the model for Linear layers in attention modules. This matches the notebook approach.

```bash
# Auto-discover modules (recommended)
python esmretrain.py --aa-fasta ... --three-di-fasta ... --lora-target-modules ""

# Or specify manually
python esmretrain.py --aa-fasta ... --three-di-fasta ... \
  --lora-target-modules "q_proj,k_proj,v_proj,o_proj,fc1,fc2"
```

### 2. Model Selection

Choose the ESM-2 model size based on your needs:

```bash
# Small & fast (35M params) - good for testing
--hf-model facebook/esm2_t12_35M_UR50D

# Medium (150M params) - balanced
--hf-model facebook/esm2_t30_150M_UR50D

# Large (650M params) - best quality
--hf-model facebook/esm2_t33_650M_UR50D
```

### 3. GPU Selection

Specify which GPU to use:

```bash
# Use GPU 0
python esmretrain.py --device cuda:0 --aa-fasta ... --three-di-fasta ...

# Use GPU 1
python esmretrain.py --device cuda:1 --aa-fasta ... --three-di-fasta ...

# Force CPU
python esmretrain.py --device cpu --aa-fasta ... --three-di-fasta ...
# or
python esmretrain.py --cpu --aa-fasta ... --three-di-fasta ...
```

### 4. Batch Size Configuration

Adjust batch size based on your GPU memory:

```bash
# Small batch for large models or limited memory
--batch-size 1

# Default
--batch-size 2

# Larger batch for faster training (if you have memory)
--batch-size 4
```

### 5. Configuration File

All parameters can be specified in a JSON config file:

```json
{
  "aa_fasta": "data/sequences.fasta",
  "three_di_fasta": "data/structures_3di.fasta",
  "hf_model": "facebook/esm2_t12_35M_UR50D",
  "mask_label_chars": "X",
  "lora_r": 8,
  "lora_alpha": 16.0,
  "lora_dropout": 0.05,
  "lora_target_modules": "",
  "batch_size": 2,
  "epochs": 10,
  "lr": 0.0001,
  "weight_decay": 0.01,
  "num_workers": 0,
  "log_every": 10,
  "device": "cuda:1",
  "out_dir": "my_checkpoints"
}
```

Then run:
```bash
python esmretrain.py --config my_config.json
```

## Training Parameters

### LoRA Parameters
- `--lora-r`: Rank of LoRA adapters (default: 8)
- `--lora-alpha`: Alpha parameter for LoRA (default: 16.0)
- `--lora-dropout`: Dropout rate for LoRA (default: 0.05)
- `--lora-target-modules`: Modules to apply LoRA (empty = auto-discover)

### Training Hyperparameters
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for optimizer (default: 1e-2)
- `--log-every`: Log progress every N steps (default: 10)

### Data Parameters
- `--mask-label-chars`: Characters to mask in 3Di sequences (default: "X")
  - These positions are ignored in loss calculation
  - Useful for low-confidence predictions from structure prediction tools

## Complete Example

```bash
python esmretrain.py \
  --aa-fasta data/train_sequences.fasta \
  --three-di-fasta data/train_3di.fasta \
  --hf-model facebook/esm2_t30_150M_UR50D \
  --device cuda:1 \
  --batch-size 4 \
  --epochs 10 \
  --lr 1e-4 \
  --lora-r 16 \
  --lora-alpha 32.0 \
  --out-dir checkpoints/run_001 \
  --mask-label-chars "X?"
```

## Output

The script saves checkpoints after each epoch in the output directory:
- `epoch_1.pt`
- `epoch_2.pt`
- `epoch_3.pt`
- etc.

Each checkpoint contains:
- Model weights (LoRA adapters + classifier)
- Label vocabulary
- Training configuration
- LoRA target modules used

## Inference

After training, use the saved checkpoint for predictions:

```python
from esmretrain import predict_3di_for_fasta

results = predict_3di_for_fasta(
    model_ckpt="checkpoints/epoch_10.pt",
    aa_fasta="new_sequences.fasta",
    device="cuda:0"
)

for header, aa_seq, pred_3di in results:
    print(f">{header}")
    print(f"AA:  {aa_seq}")
    print(f"3Di: {pred_3di}")
    print()
```

## Differences from Original Script

The updated script now:

1. **Auto-discovers LoRA target modules** by scanning the model architecture
   - Matches the notebook's dynamic approach
   - Falls back to sensible defaults if discovery fails

2. **Supports explicit GPU selection** via `--device` parameter
   - Can specify `cuda:0`, `cuda:1`, etc.
   - Useful for multi-GPU systems

3. **Allows config file loading** for easier experiment management
   - All parameters in one JSON file
   - Better reproducibility

4. **Provides better logging** of model configuration
   - Shows discovered LoRA modules
   - Prints trainable parameter counts

5. **Uses smaller default model** (`esm2_t12_35M_UR50D`)
   - Faster for initial testing
   - Can easily switch to larger models

6. **Saves LoRA target modules** in checkpoints
   - Ensures inference uses same modules as training
   - Better reproducibility

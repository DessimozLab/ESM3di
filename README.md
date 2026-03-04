# ESM3Di

ESM + PEFT LoRA for 3Di per-residue prediction. Train ESM-2 or ESM++ models with LoRA adapters to predict 3Di structural sequences from amino acid sequences.

## Features

- 🧬 Train ESM-2 and ESM++ models for 3Di structure prediction
- 🎯 Memory-efficient training using LoRA (Low-Rank Adaptation)
- 🔧 Support for masking low-confidence positions
- ⚡ Multi-GPU training with DataParallel
- 🔀 Multi-GPU inference with automatic sharding

## Installation

### Option 1: Using Conda (Recommended)

1. Create and activate the conda environment:
```bash
# For the full training environment:
conda env create -f environment.yml
conda activate esm3di

# For the inference-only environment (includes FoldSeek):
conda env create -f fastas2foldseekdb_env.yml
conda activate fastas2foldseekdb
```

### Option 2: Using pip

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Downloading Test Structures from AlphaFold Database

Download random AlphaFold structures for testing or training:

```bash
python -m esm3di.testdataset \
    --count 10 \
    --output-dir test_structures \
    --seed 42
```

Or download specific proteins by UniProt accession:

```bash
python -m esm3di.testdataset \
    --accessions P04637 P01112 P42574 \
    --output-dir structures
```

#### Download Options

- `--count`: Number of random structures to download
- `--accessions`: Specific UniProt accessions to download
- `--output-dir`: Output directory (default: alphafold_structures)
- `--delay`: Delay between downloads in seconds (default: 0.5)
- `--seed`: Random seed for reproducible sampling
- `--version`: AlphaFold model version (default: 4)

**Note**: Downloaded structures are from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)

### Building Training Dataset from PDB Files

Generate training data from PDB structures with pLDDT-based masking:

```bash
python -m esm3di.build_trainingset \
    --pdb-dir alphafold_structures/ \
    --output-prefix training_data \
    --plddt-threshold 70 \
    --mask-char X
```

This will:
1. Parse PDB files to extract sequences and pLDDT scores (from B-factor column)
2. Use FoldSeek to generate 3Di sequences
3. Mask low-confidence positions (pLDDT < threshold) in 3Di sequences
4. Output AA and masked 3Di FASTA files ready for training

#### Building Training Set Options

- `--pdb-dir`: Directory containing PDB files (or use `--pdb-files` for specific files)
- `--output-prefix`: Prefix for output files
- `--plddt-threshold`: pLDDT threshold below which to mask (default: 70)
- `--mask-char`: Character for masking low-confidence positions (default: X)
- `--split-chains`: Split multi-chain structures into separate entries
- `--chain`: Extract specific chain only

**Output files:**
- `{prefix}_aa.fasta`: Amino acid sequences
- `{prefix}_3di_masked.fasta`: 3Di sequences with masked positions
- `{prefix}_stats.txt`: Statistics about masking

### Training

Train a model using FASTA files with amino acid sequences and corresponding 3Di labels:

```bash
python -m esm3di.esmretrain \
    --aa-fasta data/sequences.fasta \
    --three-di-fasta data/3di_labels.fasta \
    --hf-model facebook/esm2_t33_650M_UR50D \
    --mask-label-chars "X" \
    --batch-size 4 \
    --epochs 10 \
    --lr 1e-4 \
    --out-dir checkpoints/
```

#### Using ESM++ Models

ESM++ models from Synthyra (via HuggingFace) are supported and offer improved performance:

```bash
# Train with ESM++ Small (333M params)
python -m esm3di.esmretrain \
    --aa-fasta data/sequences.fasta \
    --three-di-fasta data/3di_labels.fasta \
    --hf-model Synthyra/ESMplusplus_small \
    --mask-label-chars "X" \
    --batch-size 4 \
    --epochs 10 \
    --lr 2e-4 \
    --out-dir checkpoints/

# Or use ESM++ Large for better quality
python -m esm3di.esmretrain \
    --hf-model Synthyra/ESMplusplus_large \
    # ... other args
```

**Available ESM++ Models:**
- `Synthyra/ESMplusplus_small`: 333M parameters
- `Synthyra/ESMplusplus_large`: 575M parameters

ESM++ models provide:
- Better protein representations than ESM-2
- Faster inference
- Improved scaling and performance
- Native HuggingFace integration (no additional dependencies)

#### Key Arguments

- `--aa-fasta`: FASTA file with amino acid sequences
- `--three-di-fasta`: FASTA file with matching 3Di sequences (same order and length)
- `--hf-model`: Model identifier. ESM-2 options: `facebook/esm2_t12_35M_UR50D` (35M), `facebook/esm2_t30_150M_UR50D` (150M), `facebook/esm2_t33_650M_UR50D` (650M). ESM++ options: `Synthyra/ESMplusplus_small` (333M), `Synthyra/ESMplusplus_large` (575M)
- `--mask-label-chars`: Characters to treat as masked (e.g., low pLDDT positions)
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA scaling factor (default: 16.0)
- `--batch-size`: Training batch size per GPU
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--out-dir`: Directory to save checkpoints
- `--multi-gpu`: Enable multi-GPU training (uses all available GPUs)
- `--mixed-precision`: Enable FP16 mixed precision training
- `--gradient-accumulation-steps`: Accumulate gradients over N batches

### Multi-GPU Training

For training on multiple GPUs, use the `--multi-gpu` flag:

```bash
python -m esm3di.esmretrain \
    --aa-fasta data/sequences.fasta \
    --three-di-fasta data/3di_labels.fasta \
    --hf-model Synthyra/ESMplusplus_small \
    --batch-size 8 \
    --multi-gpu \
    --mixed-precision \
    --epochs 10 \
    --out-dir checkpoints/
```

This uses `torch.nn.DataParallel` to automatically distribute batches across all available GPUs. The effective batch size is `batch_size * num_gpus`.

**Multi-GPU Training Options:**
- `--multi-gpu`: Enable DataParallel multi-GPU training
- `--mixed-precision`: Enable FP16 for faster training and reduced memory
- `--gradient-accumulation-steps`: Simulate larger batches when GPU memory is limited
- `--device`: Specify primary GPU (e.g., `cuda:0`)

**Example with gradient accumulation:**
```bash
# Effective batch size = 4 * 4 * 2 GPUs = 32
python -m esm3di.esmretrain \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --multi-gpu \
    # ... other args
```

### Using Config Files

For reproducible experiments, use JSON config files:

```bash
python -m esm3di.esmretrain --config config_esmpp_large.json
```

Example config file:
```json
{
  "aa_fasta": "data/sequences.fasta",
  "three_di_fasta": "data/3di_labels.fasta",
  "hf_model": "Synthyra/ESMplusplus_small",
  "mask_label_chars": "X",
  "use_cnn_head": true,
  "batch_size": 8,
  "epochs": 3,
  "lr": 0.0002,
  "multi_gpu": true,
  "mixed_precision": true,
  "out_dir": "checkpoints_esmpp"
}
```

### Inference

Use the trained model to predict 3Di sequences:

```python
from esm3di import predict_3di_for_fasta

results = predict_3di_for_fasta(
    model_ckpt="checkpoints/epoch_10.pt",
    aa_fasta="data/test_sequences.fasta",
    device="cuda"  # or "cpu"
)

for header, aa_seq, pred_3di in results:
    print(f">{header}")
    print(f"AA:  {aa_seq}")
    print(f"3Di: {pred_3di}")
```

### Creating FoldSeek Database

Generate a FoldSeek-compatible database from amino acid sequences:

```bash
python -m esm3di.fastas2foldseekdb \
    --aa-fasta data/proteins.fasta \
    --model-ckpt checkpoints/epoch_10.pt \
    --output-db my_foldseek_db
```

This will:
1. Run ESM inference to predict 3Di sequences
2. Create intermediate AA and 3Di FASTA files
3. Build a FoldSeek database with both sequence and structure information

#### Multi-GPU Inference

For large datasets, enable multi-GPU inference to parallelize predictions:

```bash
python -m esm3di.fastas2foldseekdb \
    --aa-fasta data/large_dataset.fasta \
    --model-ckpt checkpoints/epoch_10.pt \
    --output-db my_foldseek_db \
    --multi-gpu \
    --num-gpus 4
```

**How Multi-GPU Inference Works:**
1. Input sequences are sharded across GPUs using round-robin distribution
2. Each GPU runs as an isolated subprocess with its own CUDA context
3. Predictions are merged back into original sequence order
4. FoldSeek database is built from the merged results

**Multi-GPU Inference Options:**
- `--multi-gpu`: Enable multi-GPU inference
- `--num-gpus`: Number of GPUs to use (default: all available)

#### Using Pre-computed 3Di Sequences

If you already have 3Di predictions:

```bash
python -m esm3di.fastas2foldseekdb \
    --aa-fasta data/proteins.fasta \
    --three-di-fasta data/proteins_3di.fasta \
    --output-db my_foldseek_db \
    --skip-inference
```

#### FoldSeek Database Options

- `--keep-fastas`: Keep intermediate FASTA files after database creation
- `--output-aa-fasta`: Specify path for AA FASTA output
- `--output-3di-fasta`: Specify path for 3Di FASTA output
- `--foldseek-bin`: Custom path to foldseek binary

**Note**: FoldSeek must be installed and available in your PATH. Download from [https://github.com/steineggerlab/foldseek](https://github.com/steineggerlab/foldseek)

## Data Format

### Input FASTA Files

Both amino acid and 3Di FASTA files should have:
- Matching number of sequences
- Sequences in the same order
- Equal length for corresponding AA and 3Di sequences

Example `sequences.fasta`:
```
>protein1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK
>protein2
KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIE
```

Example `3di_labels.fasta`:
```
>protein1
acbdACBDacbdACBDacbdACBDacbdACBDacbdACBDacbdACBDacbdA
>protein2
XbdACBDacbdACBDacbdACBDacbdACBDacbdACBDacbdACBDacbdACB
```

Note: Characters specified in `--mask-label-chars` (e.g., 'X') will be ignored during training.

## Model Checkpoints

Checkpoints are saved after each epoch and contain:
- Model state dict (including LoRA adapters)
- Label vocabulary
- Masked label characters
- Training arguments

### Available Pre-trained Checkpoints

| Checkpoint | Model | CNN Head | Loss | Predictions |
|------------|-------|----------|------|-------------|
| `checkpoints/epoch_3.pt` | ESM2 35M | No | ~N/A | **Working** |
| `checkpoints_mk2/epoch_3.pt` | ESM2 | Yes | 1.51 | Untested |
| `checkpoints_ESM2big/epoch_3.pt` | ESM2 | Yes | 1.51 | Untested |
| `checkpoints_esmpp_bfvd/epoch_3.pt` | ESM++ small | Yes | - | Trained on BFVD |

*Some ESM++ checkpoints may produce near-uniform predictions due to the model's layer-normalized hidden states having low variance.

**Recommended checkpoint for inference:** `checkpoints/epoch_3.pt` (ESM2 35M, no CNN head) or the BFVD-trained ESM++ checkpoint.

### Verifying Predictions

Use the test script to verify that model outputs have sufficient diversity:

```bash
python test_output_diversity.py output_3di.fasta
```

This will check that:
- Output contains multiple unique 3Di characters
- No single character dominates more than 50% of predictions
- Output is not effectively uniform (>90% one character)

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- transformers ≥ 4.30.0
- peft ≥ 0.5.0
- CUDA-capable GPU (recommended)
- biopython (for FoldSeek database creation)

**For multi-GPU training/inference:**
- Multiple CUDA-capable GPUs
- Sufficient GPU memory (ESM++ small: ~4GB per GPU, ESM++ large: ~8GB per GPU)

## License

MIT License

## Citation

If you use this code, please cite the relevant papers:
- ESM-2: [Lin et al., 2022]
- ESM++: [Synthyra, 2024] - https://huggingface.co/Synthyra
- LoRA: [Hu et al., 2021]
- 3Di: [van Kempen et al., 2023]


## Funding

Funded by NIH through the [Pathogen Data Network](https://www.pathogendatanetwork.org).

*This resource is supported by the National Institute Of Allergy And Infectious Diseases of the National Institutes of Health under Award Number U24AI183840. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

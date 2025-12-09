# ESM3Di

ESM + PEFT LoRA for 3Di per-residue prediction. Train an ESM-2 model with LoRA adapters to predict 3Di structural sequences from amino acid sequences.

## Features

- 🧬 Train ESM-2 models for 3Di structure prediction
- 🎯 Memory-efficient training using LoRA (Low-Rank Adaptation)
- 🔧 Support for masking low-confidence positions
- 📊 Per-residue token classification
- 🚀 Inference on new sequences

## Installation

### Option 1: Using Conda (Recommended)

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate esm3di
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

#### Using ESMC Models

The newer ESMC models from EvolutionaryScale are now supported and offer improved performance:

```bash
# First install the esm library
pip install esm

# Train with ESMC-300M
python -m esm3di.esmretrain \
    --aa-fasta data/sequences.fasta \
    --three-di-fasta data/3di_labels.fasta \
    --hf-model esmc-300m-2024-12 \
    --mask-label-chars "X" \
    --batch-size 4 \
    --epochs 10 \
    --lr 1e-4 \
    --out-dir checkpoints/

# Or use ESMC-600M for better quality
python -m esm3di.esmretrain \
    --hf-model esmc-600m-2024-12 \
    # ... other args
```

**Available ESMC Models:**
- `esmc-300m-2024-12`: 300M parameters, 30 layers
- `esmc-600m-2024-12`: 600M parameters, 36 layers

ESMC models provide:
- Better protein representations than ESM-2
- Faster inference
- Improved scaling and performance

**Note:** ESMC models require the `esm` library from EvolutionaryScale. Install with `pip install esm`.

#### Key Arguments

- `--aa-fasta`: FASTA file with amino acid sequences
- `--three-di-fasta`: FASTA file with matching 3Di sequences (same order and length)
- `--hf-model`: Model identifier. ESM-2 options: `facebook/esm2_t12_35M_UR50D` (35M), `facebook/esm2_t30_150M_UR50D` (150M), `facebook/esm2_t33_650M_UR50D` (650M). ESMC options: `esmc-300m-2024-12` (300M), `esmc-600m-2024-12` (600M)
- `--mask-label-chars`: Characters to treat as masked (e.g., low pLDDT positions)
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA scaling factor (default: 16.0)
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--out-dir`: Directory to save checkpoints

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

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- transformers ≥ 4.30.0
- peft ≥ 0.5.0
- CUDA-capable GPU (recommended)
- esm (optional, for ESMC models): `pip install esm`

For ESMC model support, install the EvolutionaryScale ESM library:
```bash
pip install esm
```

This enables training with the newer ESMC-300M and ESMC-600M models.

## License

MIT License

## Citation

If you use this code, please cite the relevant papers:
- ESM-2: [Lin et al., 2022]
- LoRA: [Hu et al., 2021]
- 3Di: [van Kempen et al., 2023]

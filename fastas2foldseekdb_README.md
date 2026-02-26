# fastas2foldseekdb

Generate FoldSeek databases from amino acid FASTA files using ESM 3Di predictions.

## Setup

### 1. Create the conda environment

```bash
conda env create -f fastas2foldseekdb.yml
conda activate fastas2foldseekdb
```

On first activation, FastPLMs will be automatically cloned to the environment and PYTHONPATH will be configured.

## Usage

### Basic usage with inference

```bash
python fastas2foldseekdb.py \
    --aa-fasta path/to/proteins.fasta \
    --model-ckpt path/to/checkpoints/epoch_3.pt \
    --output-db path/to/output_db \
    --device cuda:0
```

### Using pre-computed 3Di FASTA (skip inference)

```bash
python fastas2foldseekdb.py \
    --aa-fasta path/to/proteins.fasta \
    --three-di-fasta path/to/proteins_3di.fasta \
    --output-db path/to/output_db \
    --skip-inference
```

### Full options

| Option | Description |
|--------|-------------|
| `--aa-fasta` | Input amino acid FASTA file (required) |
| `--three-di-fasta` | Pre-computed 3Di FASTA file |
| `--model-ckpt` | Path to trained ESM 3Di model checkpoint |
| `--skip-inference` | Skip inference, use provided 3Di FASTA |
| `--device` | Device for inference (cuda:0, cuda:1, cpu) |
| `--output-db` | Output FoldSeek database path (required) |
| `--keep-fastas` | Keep intermediate AA and 3Di FASTA files |
| `--output-aa-fasta` | Path to save AA FASTA |
| `--output-3di-fasta` | Path to save 3Di FASTA |
| `--foldseek-bin` | Path to foldseek binary (default: foldseek) |

## Output

The pipeline produces a FoldSeek database with the following files:

- `{output_db}` - Main amino acid database
- `{output_db}_ss` - 3Di structure database  
- `{output_db}_h` - Header database
- `{output_db}.fasta` - Amino acid FASTA
- `{output_db}_3di.fasta` - Predicted 3Di FASTA (if `--keep-fastas`)
- `{output_db}_aa.fasta` - Copy of input AA FASTA (if `--keep-fastas`)

## Example

```bash
# Process alphacoronavirus sequences
python fastas2foldseekdb.py \
    --aa-fasta ./alphacorona/alphacoronaviridae.fasta \
    --model-ckpt ./checkpoints_ESMplusplus_small/epoch_3.pt \
    --output-db ./alphacorona/alphacoronaviridae \
    --keep-fastas \
    --device cuda:1
```

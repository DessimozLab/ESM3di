# ESM3Di Copilot Instructions

## Project Overview
ESM3Di trains ESM-2 and ESM++ protein language models with LoRA adapters to predict **3Di structural sequences** from amino acid sequences. The 3Di alphabet (from FoldSeek) encodes local protein 3D structure into a 20-character alphabet, enabling sequence-based structural comparison.

## Architecture

### Core Components
- **`esm3di/ESM3di_model.py`**: Main model wrapper (`ESM3DiModel`) with LoRA setup, CNN classification head, and inference logic
- **`esm3di/esmretrain.py`**: Training script with multi-GPU support, mixed precision, and TensorBoard logging
- **`esm3di/fastas2foldseekdb.py`**: Inference → FoldSeek database pipeline with multi-GPU sharding
- **`esm3di/build_trainingset.py`**: PDB → training FASTA conversion with pLDDT-based masking

### Data Flow
1. **Training**: AA FASTA + 3Di FASTA → `Seq3DiDataset` → tokenize → model → per-residue classification
2. **Inference**: AA FASTA → model prediction → 3Di FASTA → FoldSeek database

### Key Classes
```python
ESM3DiModel           # Main wrapper: handles LoRA injection, model loading
ESMWithCNNHead        # Wraps base model with CNN classification head
CNNClassificationHead # 1D CNN for per-token classification
Seq3DiDataset         # Paired AA/3Di FASTA dataset
```

## Development Commands

```bash
# Install in development mode
pip install -e .

# Training (use config file)
python -m esm3di.esmretrain --config config_example.json

# Quick training test
python -m esm3di.esmretrain --aa-fasta test_data_aa.fasta --three-di-fasta test_data_3di.fasta --epochs 1 --batch-size 2

# Build training data from PDBs
python -m esm3di.build_trainingset --pdb-dir structures/ --output-prefix train_data --plddt-threshold 70

# Create FoldSeek database
python -m esm3di.fastas2foldseekdb --aa-fasta proteins.fasta --model-ckpt checkpoints/epoch_3.pt --output-db my_db

# Multi-GPU inference
python -m esm3di.fastas2foldseekdb --aa-fasta large.fasta --model-ckpt ckpt.pt --output-db db --multi-gpu --num-gpus 4

# Download test structures from AlphaFold
python -m esm3di.testdataset --count 10 --output-dir test_structures
```

## Code Patterns

### Model Loading Pattern
```python
esm_model = ESM3DiModel(
    hf_model_name="Synthyra/ESMplusplus_small",  # or facebook/esm2_t33_650M_UR50D
    num_labels=len(label_vocab),
    lora_r=8, lora_alpha=16.0,
    use_cnn_head=True
)
model = esm_model.get_model()
```

### Checkpoint Structure
Checkpoints contain: `model_state_dict`, `label_vocab`, `mask_label_chars`, `args`, `epoch`, `optimizer_state_dict`

### LoRA Target Module Discovery
`discover_lora_target_modules()` auto-discovers Linear/Embedding layers, excluding classifier heads. Prefer auto-discovery over hardcoded module names.

### 3Di Vocabulary
- Default 20 characters: `a-t` (case-insensitive in data, uppercase in processing)
- `X` = masked position (low pLDDT confidence, ignored in loss via `mask_label_chars`)

## Conventions

### FASTA Handling
- Use `read_fasta()` from `ESM3di_model.py` for simple parsing
- AA and 3Di FASTAs must have **matching order and equal sequence lengths**
- Headers must match between paired files

### Tokenization
- ESM2: Use `AutoTokenizer.from_pretrained()`
- ESM++: Use `model.tokenizer` (built-in)
- Always use `trust_remote_code=True` for HuggingFace loading

### Multi-GPU
- Training: uses `torch.nn.DataParallel` (wrap model after LoRA setup)
- Inference: uses subprocess isolation with round-robin sequence sharding

### Config Files
JSON config files override CLI args. Keys use underscores: `batch_size`, `three_di_fasta`, `hf_model`.

## Testing

```bash
# Verify output diversity (check predictions aren't degenerate)
python test_output_diversity.py output_3di.fasta
```

## External Dependencies
- **FoldSeek**: Required for database creation and 3Di generation. Must be in PATH.
- **HuggingFace Models**: `facebook/esm2_*`, `Synthyra/ESMplusplus_*`
- **Pre-trained checkpoints**: `cactuskid13/esm2small_3di`, `cactuskid13/ESMpp_small_3Di` on HuggingFace Hub

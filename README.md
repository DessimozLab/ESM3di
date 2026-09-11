# ESM3Di

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%22.0%2B-ee4c2c.svg)](https://pytorch.org/)

**ESM3Di** predicts 3D interaction (3Di) structural alphabets directly from primary amino acid sequences using fine-tuned ESM models (specifically optimized for viral protein structures). By bypassing explicit 3D atomic coordinate prediction, `esm3di` enables ultra-fast structural alignment, database construction for [Foldseek](https://github.com/steineggerlab/foldseek), and automated phylogenetic tree generation at scale.

---

## ✨ Features

- **Direct Sequence-to-3Di Prediction:** Translates protein amino acid FASTA inputs to Foldseek 3Di strings in seconds.
- **Native Foldseek Integration:** Compiles structural databases (`.db` files) directly without requiring PDB/mmCIF generation.
- **Per-Residue Model Confidence:** Computes token-level perplexity scores to flag uncertain regions or low-confidence predictions.
- **End-to-End Structural Phylogenetics (`foldtree`):** Automated Snakemake workflow for 3Di prediction, structural alignment, and phylogenetic tree reconstruction (supports both ESM3Di and ProsTT5 backends).
- **Scalable Hardware Execution:** Multi-GPU parallel batched processing with automatic CPU fallback.

---

## 🛠️ Installation

### Prerequisites

- **OS:** Linux, macOS, or Windows
- **Python:** `≥ 3.9`
- **PyTorch:** `≥ 2.0`

### 📥 Installation

```bash
# Clone the repository
git clone [https://github.com/DessimozLab/ESM3di.git](https://github.com/DessimozLab/ESM3di.git)
cd ESM3di

# Create and activate environment
conda create -n esm3di python=3.10 -y
conda activate esm3di

# Install ESM3Di in editable mode
pip install -e .
```

> **Note on GPU Acceleration:** Standard `pip install -e .` pulls the default PyTorch wheel. If your GPU cluster requires a specific CUDA toolkit version (e.g., CUDA 12.1), pre-install PyTorch via the [official PyTorch guide](https://pytorch.org/get-started/locally/) before running `pip install -e .`.

---

## 🚀 Quick Start

Run 3Di predictions, build databases, or reconstruct phylogenetic trees:

```bash
# 1. Predict 3Di sequences
esm3di predict --input-fasta test_data/test_virus.fasta --output-fasta outputs/output_3di.fasta

# 2. Build a Foldseek database directly
esm3di foldseek-db --input-fasta test_data/test_virus.fasta --output-db outputs/foldseek_db

# 3. Run full phylogenetic tree inference (FoldTree pipeline)
esm3di foldtree -i test_data/test_virus.fasta -o results --cores 8
```

---

## 💻 Command Line Interface (CLI)

`esm3di` provides four subcommands: `predict`, `foldseek-db`, `perplexity`, and `foldtree`.

```text
usage: esm3di [-h] {predict,foldseek-db,perplexity,foldtree} ...

positional arguments:
  {predict,foldseek-db,perplexity,foldtree}
    predict             Predict 3Di sequences from an amino acid FASTA file and save to FASTA.
    foldseek-db         Predict 3Di sequences and compile directly into a Foldseek-compatible database.
    perplexity          Calculate model confidence (perplexity) for each residue position and export to TSV.
    foldtree            Run end-to-end 3Di prediction and phylogenetic tree inference via Snakemake.
```

---

### Common Options

The following flags are available across all subcommands:

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model-ckpt` | `str` | [`cactuskid13/ESM3di_Small_MLM_3di`](https://huggingface.co/cactuskid13/ESM3di_Small_MLM_3di/tree/main/hf_compatible) | Hugging Face repository ID or local checkpoint path or  |
| `--num-gpus` | `int` | `None` | Number of GPUs to use (default: use all available). |
| `--revision` | `str` | `46c5f7d` | Hugging Face model revision or commit SHA. |
| `--batch-size` | `int` | `4` | Inference batch size per device. |

---

### 1. `predict` — Generate 3Di FASTA

Translates amino acid sequences into matching 3Di structural sequences saved in FASTA format.

```bash
esm3di predict \
  --input-fasta test_data/test_virus.fasta \
  --output-fasta outputs/output_3di.fasta \
  --batch-size 8
```

**Subcommand Flags:**

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--input-fasta` | `str` | `test_data/example_input.fasta` | Input protein amino acid FASTA file. |
| `--output-fasta` | `str` | `outputs/output_3di.fasta` | Destination path for output 3Di FASTA file. |

---

### 2. `foldseek-db` — Build Foldseek Database

Runs sequence prediction and automatically formats output into a binary Foldseek structure database ready for alignment searches.

```bash
esm3di foldseek-db \
  --input-fasta test_data/test_virus.fasta \
  --output-db outputs/foldseek_db
```

**Subcommand Flags:**

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--input-fasta` | `str` | `test_data/example_input.fasta` | Input protein amino acid FASTA file. |
| `--output-db` | `str` | `outputs/foldseek_db` | Prefix path for output Foldseek database files. |

---

### 3. `perplexity` — Per-Residue Confidence Metrics

Calculates token confidence/perplexity scores for each amino acid position across sequences and exports to TSV.

```bash
esm3di perplexity \
  --input-fasta test_data/test_virus.fasta \
  --output-tsv outputs/output_confidence.tsv
```

**Subcommand Flags:**

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--input-fasta` | `str` | `test_data/example_input.fasta` | Input protein amino acid FASTA file. |
| `--output-tsv` | `str` | `outputs/output_confidence.tsv` | Target path to export TSV metrics file. |

---

### 4. `foldtree` — End-to-End Phylogenetic Tree Inference

Executes an embedded Snakemake workflow to predict 3Di states, align structural sequences, and build rooted phylogenetic trees.

```bash
# Run standard ESM3Di FoldTree workflow
esm3di foldtree -i test_data/test_virus.fasta -o results --cores 8

# Run dry-run to preview execution steps
esm3di foldtree -i test_data/test_virus.fasta -o results -n

# Use ProsTT5 backend instead of ESM3Di
esm3di foldtree -i test_data/test_virus.fasta -o results --use-prostt5
```

**Subcommand Flags:**

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `-i`, `--input-fasta` | `str` | `test_data/test_virus.fasta` | Input FASTA file or directory containing FASTA files. |
| `-o`, `--output-dir` | `str` | `results` | Directory to save phylogenetic trees and intermediate files. |
| `-d`, `--dataset` | `str` | `None` | Dataset identifier prefix (defaults to input file stem). |
| `-c`, `--cores` | `int` | `4` | Number of CPU cores for Snakemake execution. |
| `-n`, `--dry-run` | `flag` | `False` | Print execution plan and rules without executing. |
| `--use-prostt5` | `flag` | `False` | Use ProstT5 instead of ESM3Di for 3Di prediction. |

---

## 🐍 Python API Usage

```python
import logging
from esm3di.inference import ESM3DiPredictor
from esm3di.io import fasta2foldseek

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Initialize predictor
predictor = ESM3DiPredictor.from_pretrained("DessimozLab/esm3di", revision="46c5f7d")

# 1. In-Memory Sequence Prediction
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
p_3di = predictor.predict(sequence)
print(f"3Di Output: {p_3di}\n")

# 2. FASTA Translation & Foldseek Database Compilation
predictor.predict_fasta("test_data/test_virus.fasta", "outputs/output_3di.fasta", batch_size=4)
fasta2foldseek(
    aa_input="test_data/test_virus.fasta",
    tdi_input="outputs/output_3di.fasta",
    output_basename="outputs/foldseek_db"
)

# 3. Per-Residue Perplexity Calculation
predictor.output_per_position_perplexity(
    input_fasta_path="test_data/test_virus.fasta",
    output_tsv_path="outputs/output_confidence.tsv",
    batch_size=16
)
```

---

## 📁 Repository Structure

```text
ESM3di/                                # Repository Root
├── src/                               # Python source directory
│   └── esm3di/                        # Main package
│       ├── __init__.py                # Package entry point
│       ├── cli.py                     # CLI router (predict, foldseek-db, perplexity, foldtree)
│       ├── model.py                   # Model architecture definitions
│       ├── inference.py               # Core inference engine & predictor API
│       ├── io.py                      # File I/O and Foldseek DB formatting
│       ├── preprocessing.py           # Multi-GPU sequence sharding utilities
│       │
│       └── workflows/                 # Embedded Viral-FoldTree Workflow
│           ├── __init__.py
│           ├── Snakefile              # Snakemake workflow entry point
│           ├── config.yaml            # Default runtime parameters
│           ├── rules/                 # Modular Snakemake rules (.smk)
│           ├── envs/                  # Conda environment definitions
│           └── scripts/               # Helper scripts executed by rules
│
├── tests/                             # Unit & integration tests
│   ├── test_model.py
│   └── test_workflow.py               # Validates Snakemake dry-run pipeline
│
├── test_data/                         # Test inputs tracked in Git
│   └── test_virus.fasta               # Verification sequence file
│
├── checkpoints/                       # Heavy Model Weights
│   └── hf_compatible/                 # Tracked via Git LFS
│
├── short_notice_results/              # Paper figures & benchmarks
│   ├── data/
│   └── run_paper_figures.sh           # Script to recreate manuscript figures
│
├── research_archive/                  # Historical experimental files
│   └── legacy_readme.md               # Guide to old research experiments
│
├── MANIFEST.in                        # Package distribution directives
├── .gitignore                         # Output exclusion patterns
├── README.md                          # Primary user documentation
└── pyproject.toml                     # Python build & dependency configuration
```

---

## 📜 Citation & Credits

If you use **ESM3Di** or the **FoldTree** pipeline in your research, please cite:

```bibtex
@article{esm3di2026,
  title={ESM3Di: Direct Structural 3Di Alphabets Prediction via Evolutionary Scale Modeling},
  author={...},
  journal={Bioinformatics / GitHub Repository},
  year={2026},
  publisher={Dessimoz Lab}
}
```

This software builds upon structural 3Di representations established by [Foldseek](https://github.com/steineggerlab/foldseek).

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
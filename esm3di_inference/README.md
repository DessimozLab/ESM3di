# ESM3Di

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%22.0%2B-ee4c2c.svg)](https://pytorch.org/)

**ESM3Di** predicts 3D interaction (3Di) structural alphabets directly from primary amino acid sequences using fine-tuned ESM models. By bypassing explicit 3D atomic coordinate prediction, `esm3di` enables ultra-fast structural alignment and database construction for [Foldseek](https://github.com/steineggerlab/foldseek) at scale.

---

## ✨ Features

- **Direct Sequence-to-3Di Prediction:** Translates protein amino acid FASTA inputs to Foldseek 3Di strings in seconds.
- **Native Foldseek Integration:** Direct compilation of structural databases (`.db` files) without needing PDB/mmCIF generation.
- **Per-Residue Model Confidence:** Computes per-token perplexity scores to flag uncertain regions or low-confidence predictions.
- **Scalable Execution:** Supports single-CPU testing as well as multi-GPU parallel batched processing across available hardware accelerators.

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

# Install ESM3Di
pip install -e .
```
> **Note on GPU Acceleration:** Standard `pip install -e .` will pull the default PyTorch wheel. If your GPU cluster requires a specific CUDA toolkit version (e.g., CUDA 12.1), pre-install PyTorch via the [official PyTorch guide](https://pytorch.org/get-started/locally/) prior to running `pip install -e .`.

---

## 🚀 Quick Start

Run 3Di predictions on an example FASTA file:

```bash
# Predict 3Di sequences
esm3di predict --input-fasta example_input.fasta --output-fasta outputs/output_3di.fasta

# Build a Foldseek database directly
esm3di foldseek --input-fasta example_input.fasta --output-db outputs/foldseek_db

```

---

## 💻 Command Line Interface (CLI)

`esm3di` provides three core subcommands: `predict`, `foldseek`, and `perplexity`.

```text
usage: esm3di [-h] {predict,foldseek,perplexity} ...

positional arguments:
  {predict,foldseek,perplexity}
    predict             Predict 3Di structural sequences from an amino acid FASTA file and save as FASTA.
    foldseek            Predict 3Di sequences and compile directly into a Foldseek-compatible database.
    perplexity          Calculate model confidence (perplexity) for each residue position and export to TSV.

```

---

### 1. `predict` — Generate 3Di FASTA

Translates amino acid sequences into matching 3Di structural sequences.

```bash
esm3di predict \
  --input-fasta example_input.fasta \
  --output-fasta outputs/output_3di.fasta \
  --batch-size 4

```

**Options:**

| Flag | Type | Default                    | Description |
| --- | --- |----------------------------| --- |
| `--input-fasta` | `str` | `example_input.fasta`      | Input protein amino acid FASTA file. |
| `--output-fasta` | `str` | `outputs/output_3di.fasta` | Destination path for output 3Di FASTA file. |
| `--model-ckpt` | `str` | *[Default Weights]*        | Optional path to local checkpoint or Hugging Face repo ID. |
| `--batch-size` | `int` | `4`                        | Inference batch size per device. |
| `--num-gpus` | `int` | `None`                     | Number of GPUs to use (default: use all available). |

---

### 2. `foldseek` — Build Foldseek Database

Runs sequence prediction and automatically formats output into a binary Foldseek structure database ready for immediate alignment searches.

```bash
esm3di foldseek \
  --input-fasta example_input.fasta \
  --output-db outputs/foldseek_db

```

**Options:**

| Flag | Type | Default               | Description |
| --- | --- |-----------------------| --- |
| `--input-fasta` | `str` | `example_input.fasta` | Input protein amino acid FASTA file. |
| `--output-db` | `str` | `outputs/foldseek_db` | Prefix path for output Foldseek database files. |
| `--model-ckpt` | `str` | *[Default Weights]*   | Optional path to local checkpoint or Hugging Face repo ID. |
| `--batch-size` | `int` | `4`                   | Inference batch size per device. |
| `--num-gpus` | `int` | `None`                | Number of GPUs to use (default: use all available). |

---

### 3. `perplexity` — Per-Residue Confidence Metrics

Calculates token confidence/perplexity scores for each amino acid position across sequences.

```bash
esm3di perplexity \
  --input-fasta example_input.fasta \
  --output-tsv outputs/output_confidence.tsv

```

**Options:**

| Flag | Type | Default                         | Description |
| --- | --- |---------------------------------| --- |
| `--input-fasta` | `str` | `example_input.fasta`           | Input protein amino acid FASTA file. |
| `--output-tsv` | `str` | `outputs/output_confidence.tsv` | Target path to export TSV metrics file. |
| `--model-ckpt` | `str` | *[Default Weights]*             | Optional path to local checkpoint or Hugging Face repo ID. |
| `--batch-size` | `int` | `4`                             | Inference batch size. |

---

### Python API Usage

```python
import logging
from esm3di.inference import ESM3DiPredictor
from esm3di.io import fasta2foldseek

# Enable logging output for standard Python scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Initialize predictor
predictor = ESM3DiPredictor.from_pretrained()

# 1. In-Memory Sequence Prediction
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
p_3di = predictor.predict(sequence)
print(f"3Di Output: {p_3di}\n")

# 2. FASTA to 3Di FASTA & Foldseek Database
predictor.predict_fasta("example_input.fasta", "outputs/output_3di.fasta", batch_size=4)
fasta2foldseek(
    aa_input="example_input.fasta",
    tdi_input="outputs/output_3di.fasta",
    output_basename="outputs/foldseek_db"
)

# 3. Per-Residue Perplexity Assessment
predictor.output_per_position_perplexity(
    input_fasta_path="example_input.fasta",
    output_tsv_path="outputs/output_confidence.tsv",
    batch_size=16
)
```

---

## 📁 Repository Structure

```text
esm3di_inference/
├── checkpoints/
│   └── hf_compatible/       # Pre-trained ESM3Di weights (Git LFS)
├── src/
│   └── esm3di/
│       ├── __init__.py      # Package entry point
│       ├── cli.py           # Command-line interface router
│       ├── inference.py     # Core inference engine & API predictor
│       ├── io.py            # File I/O and Foldseek DB formatting
│       ├── model.py         # Neural network architecture definitions
│       └── preprocessing.py # Sequence sharding utilities for multi-GPU
├── example_input.fasta      # Example amino acid FASTA file
├── pyproject.toml           # Package configuration & dependencies
└── README.md                # Project documentation
```

---

## 📜 Citation & Credits

If you use **ESM3Di** in your research, please cite:

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

```

```

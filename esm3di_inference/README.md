# ESM3Di Toolkit

The **ESM3Di Toolkit** is a production-grade bioinformatics command-line utility designed to generate structural **3Di sequence representations** directly from raw amino acid FASTA records. 

By leveraging low-rank adaptation (**LoRA**) layered over the state-of-the-art **ESM++ Small** (`Synthyra/ESMplusplus_small`) transformer backbone, the model maps complex sequence embeddings straight to a 20-character structural 3Di alphabet. These representations enable lightning-fast tertiary structure search functionality natively compatible with tools like **FoldSeek**.

## Features

- **High-Throughput 3Di Predictions:** Generate structural alphabets from standard amino acid sequences.
- **FoldSeek Database Generation:** Automatically compile protein inputs into instantly searchable binary FoldSeek layouts.
- **Per-Residue Perplexity Analytics:** Extract Shannon entropy metrics down to individual token positions for confidence scoring.
- **Asynchronous Scalability:** Automated horizontal parallel multi-GPU data sharding for high-volume sequence streaming.

---

## Installation

Ensure your localized Python virtualization virtual environment satisfies `python >= 3.9`.

```bash
# Clone the repository
git clone [https://github.com/your-username/esm3di.git](https://github.com/your-username/esm3di.git)
cd esm3di

# Initialize weights file dependencies using Git LFS
git lfs install
git lfs pull

# Install the toolkit in editable mode
pip install -e .
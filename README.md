# ESM3Di

Predict per-residue **3Di structural tokens** (the FoldSeek structural alphabet) directly from amino-acid sequences, so you can build FoldSeek databases and run structure-aware searches for proteins without structures.

The supported recipe is a two-stage fine-tune of an [ESM++](https://huggingface.co/Synthyra) protein language model:

1. **MLM stage (unsupervised).** Continue masked-language-model training of ESM++ on sequences from your domain, for example viral proteins.
2. **3Di stage (supervised).** Start from the MLM backbone, add LoRA adapters and a small CNN token-classification head, and train them to predict the 3Di token at each residue. The labels come from structures via FoldSeek, and the loss can be weighted by pLDDT.

Everything else that was tried (ESM-2/T5 backbones, iterative heads, auxiliary tracks, taxon balancing, …) lives in [`esm3di/experimental/`](esm3di/experimental/) and is kept for reference only.

```
structures (PDB/mmCIF, AlphaFold, ...) ──► esm3di prepare ──► data/{train,val,test}_{aa,3di,plddt_bins}.fasta
                                                                      │
extra sequences (optional) ─────────────────────────┐                 │
                                                    ▼                 ▼
Synthyra/ESMplusplus_{small,large} ──► esm3di mlm ──► mlm/best ──► esm3di train ──► 3di/best.pt
                                                                                        │
proteins.fasta ─────────────────────────────────────────────────────► esm3di predict ──► 3Di FASTA / FoldSeek DB
```

## Installation

```bash
# Conda (recommended)
conda env create -f environment.yml             # most GPUs
conda env create -f environment_blackwell.yml   # Blackwell GPUs (RTX 50xx, RTX PRO)
conda activate esm3di                           # or esm3di_blackwell
pip install -e .

# or plain pip
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

[FoldSeek](https://github.com/steineggerlab/foldseek) is required to extract 3Di labels from structures (`esm3di prepare`) and to build FoldSeek databases. Put it on `PATH`, or pass `--foldseek-bin /path/to/foldseek` (or set `FOLDSEEK_BIN`).

> Keep `transformers` (`>=4.45,<4.49`) and `peft` (`>=0.12,<0.14`) in the ranges given in [requirements.txt](requirements.txt). ESM++ is loaded with `trust_remote_code`, and ESM3Di pins the remote-code revision that works with these versions automatically. Override it with `--hf-revision` if you need to.

After installation the `esm3di` command is available. Without installing, use `python -m esm3di.cli` instead.

## Quick start: one command

Point the pipeline at a set of structures, and optionally at a set of extra sequences for the MLM stage:

```bash
esm3di pipeline \
  --structures structures/ \
  --sequences viral_proteins.fasta \
  --init-from Synthyra/ESMplusplus_small \
  --out-dir runs/my_model
```

The pipeline:

1. Runs `esm3di prepare` on the structures and writes `runs/my_model/data/`.
2. Runs the MLM stage on `--sequences` (plus the structure sequences if `--mlm-on-structure-sequences` is given) and writes `runs/my_model/mlm/`.
3. Trains the 3Di tagger from the MLM backbone and writes `runs/my_model/3di/best.pt`.

| You have… | Command |
|---|---|
| structures + extra sequences (full recipe) | `--structures S --sequences F.fasta` |
| only structures, but still want an MLM stage | `--structures S --mlm-on-structure-sequences` |
| only structures, no MLM stage | `--structures S` |
| already-prepared data | `--data-dir runs/old/data` instead of `--structures` |
| an MLM checkpoint already | `--init-from runs/mlm/best` (skip `--sequences`) |
| a trained 3Di checkpoint to fine-tune further | `--init-from model.pt` (no MLM stage allowed) |

Other useful options:

- Epochs, batch size, and learning rate per stage: `--mlm-epochs/--mlm-batch-size/--mlm-lr` and `--epochs/--batch-size/--lr`.
- Full config files per stage: `--mlm-config configs/mlm_esmpp_small.json` and `--train-config configs/3di_esmpp_small_from_mlm.json`.
- Splitting: `--cluster-split`, `--val-fraction`, and `--test-fraction`.
- Interrupted runs: re-run the same command with `--resume`. Prepared data is reused, a finished MLM stage is skipped, and 3Di training resumes from its latest epoch.

Every stage is also available as its own command, as described in the next section.

## Step by step

### 1. Prepare training data from structures

```bash
esm3di prepare --structures structures/ more/*.cif.gz --out-dir data [--cluster-split]
```

`--structures` accepts directories, files (`.pdb`, `.cif`, `.ent`, optionally `.gz`), or text files listing paths. Every chain becomes one example. Single-chain files are named after the file stem; multi-chain files use `stem_A`, `stem_B`, and so on. Output:

```
data/
  {all,train,val,test}_aa.fasta          amino-acid sequences
  {all,train,val,test}_3di.fasta         3Di labels (same headers/lengths)
  {all,train,val,test}_plddt_bins.fasta  pLDDT bins 0-9 per residue (from CA B-factors)
  manifest.json
```

Use `--no-plddt` for experimental structures, where B-factors are not pLDDT. `--cluster-split` assigns whole FoldSeek clusters to a single split, so the validation and test sets are not near-duplicates of the training set.

You can also bring your own paired FASTAs. They must have identical headers in the same order and equal sequence lengths. 3Di labels use the 20 letters `A-T`, case-insensitive. An optional pLDDT-bin FASTA uses digits `0-9` per residue.

### 2. MLM fine-tuning (unsupervised)

```bash
esm3di mlm --input-fasta data/train_aa.fasta extra_sequences.fasta \
           --val-fasta data/val_aa.fasta \
           --model-name Synthyra/ESMplusplus_small \
           --output-dir runs/mlm_small --epochs 3 --batch-size 4
# or: esm3di mlm --config configs/mlm_esmpp_small.json
```

The MLM stage trains all weights by default; `--use-lora true` trains adapters instead. It runs in bf16 autocast by default, and `--precision fp32` switches to full fp32. Outputs are HF model directories `runs/mlm_small/{epoch_N,best,final}/`. `best` is chosen on validation loss and perplexity. Resume an interrupted run with `--resume auto`.

### 3. 3Di training (supervised)

```bash
esm3di train --data-dir data --init-from runs/mlm_small/best --out-dir runs/3di_small
# or: esm3di train --config configs/3di_esmpp_small_from_mlm.json
```

`--init-from` accepts:

| Value | Meaning |
|---|---|
| `Synthyra/ESMplusplus_small` / `_large` (default: small) | start from the stock base model |
| an MLM output dir (`runs/mlm_small/best`) | load the MLM-tuned backbone, then add fresh LoRA and head |
| a 3Di checkpoint (`model.pt`) | continue fine-tuning a trained tagger on new data. The model architecture comes from the checkpoint; optimiser and epoch counter start fresh |

`--resume runs/3di_small/last.pt` (or `--resume auto` with `--out-dir`) continues an interrupted run with its optimiser, scheduler, and epoch counter.

Defaults follow the recipe that worked:

- LoRA r=16 / α=32 on all attention and FFN projections (auto-discovered).
- A 2-layer CNN head.
- AdamW lr 1e-4 with warmup and cosine decay.
- Mixed precision.
- pLDDT-weighted cross-entropy whenever a pLDDT-bin FASTA is available (`--plddt-min-bin` and `--plddt-weight-exponent` tune it).

Without a separate validation set, `--val-split` holds out a fraction of the training data. Multi-GPU training uses `--multi-gpu true` (DataParallel).

Each run directory contains:

```
runs/3di_small/
  epoch_N.pt  best.pt  last.pt     checkpoints (best = lowest validation loss)
  metrics.jsonl                    per-epoch train/val loss and accuracy
  train_config.json                fully resolved configuration
```

### 4. Predict 3Di and build a FoldSeek database

```bash
esm3di predict --checkpoint runs/3di_small/best.pt -i proteins.fasta -o proteins_3di.fasta \
               [--foldseek-db dbs/proteins] [--gpus 0,1] [--confidence-output conf.fasta]

# FoldSeek DB from existing AA + 3Di FASTAs
esm3di foldseekdb --aa-fasta proteins.fasta --three-di-fasta proteins_3di.fasta -o dbs/proteins
foldseek easy-search query_db dbs/proteins hits.m8 tmp --exhaustive-search 1   # example search
```

Sequences longer than the model window are predicted in overlapping windows. `esm3di-makefoldseekdb` / `python -m esm3di.fastas2foldseekdb` (see [fastas2foldseekdb_README.md](fastas2foldseekdb_README.md)) does prediction and DB creation in one step.

To sanity-check predictions, run `python -m esm3di.test_output_diversity proteins_3di.fasta`. It warns if the output collapses to a few tokens.

## Configuration files

Both trainers accept `--config file.json`. Keys are the long option names with underscores (`batch_size`, `init_from`, …), and keys starting with `_` are treated as comments. Settings are applied in this order, each overriding the previous: built-in defaults, then the config file, then `--data-dir` paths, then explicit command-line flags. Relative paths are resolved against the current working directory.

| Config | Purpose |
|---|---|
| [configs/mlm_esmpp_small.json](configs/mlm_esmpp_small.json), [configs/mlm_esmpp_large.json](configs/mlm_esmpp_large.json) | MLM stage on `data/` |
| [configs/3di_esmpp_small_from_mlm.json](configs/3di_esmpp_small_from_mlm.json), [configs/3di_esmpp_large_from_mlm.json](configs/3di_esmpp_large_from_mlm.json) | 3Di stage from the MLM output |
| [configs/bfvd/](configs/bfvd/) | Same recipe on the BFVD viral structure splits (absolute paths for our cluster) |

Older experimental options (focal loss, iterative heads, T5, auxiliary tracks, …) are rejected with a pointer to `python -m esm3di.experimental.esmretrain_full`.

## Checkpoints and compatibility

**3Di checkpoint (`.pt`).** A `torch.save` dict with `model_state_dict` (complete model: backbone, LoRA, and head), `label_vocab`, `mask_label_chars`, `args`, `epoch`, `optimizer_state_dict`, and, for new runs, a `provenance` record of which MLM checkpoint the backbone came from. The architecture is rebuilt from `args`, so prediction only needs the `.pt` file.

**MLM checkpoint (directory).** A standard HuggingFace model directory, plus `esm3di_mlm.json` with the training metadata. Directories containing only a PEFT adapter are merged into their base model on load.

**Compatibility with older checkpoints:**

- Checkpoints written by the old `esmretrain.py` / `esmretrain_mlm.py` (including `checkpoints/esm3di_from_mlm/*.pt`, the `esmc_viral_ft_large` MLM dirs, and the Hub models below) load in `esm3di predict` and `esm3di train --init-from`. The `DataParallel` `module.` prefix is handled.
- Checkpoints from older PEFT versions that wrapped extra modules with LoRA are merged into dense weights on load, and their outputs are unchanged.
- `python -m esm3di.esmretrain` and `python -m esm3di.esmretrain_mlm` still work. They forward to the new trainers, or to the experimental full versions if you pass experimental flags.

> ⚠️ **Note on earlier "MLM → 3Di" runs.** In the previous trainer, the MLM checkpoint given to the 3Di stage was not actually loaded into the backbone, so those 3Di models were trained on the stock ESM++ weights. The new trainer copies the MLM backbone before adding LoRA and logs how many tensors were transferred. Retrain with the new pipeline to get the real benefit of the MLM stage.

### Pre-trained models

| HuggingFace repository | Backbone | Notes |
|---|---|---|
| [cactuskid13/ESMpp_small_3Di](https://huggingface.co/cactuskid13/ESMpp_small_3Di) | ESM++ small | trained on BFVD (`epoch_3.pt`) |
| [cactuskid13/esm2small_3di](https://huggingface.co/cactuskid13/esm2small_3di) | ESM-2 35M | small and fast, legacy |

```bash
huggingface-cli download cactuskid13/ESMpp_small_3Di epoch_3.pt --local-dir models/
esm3di predict --checkpoint models/epoch_3.pt -i proteins.fasta -o proteins_3di.fasta
```

Upload your own model with `esm3di-push-to-hub --repo-id org/name --local-file runs/3di_small/best.pt`. MLM runs can push automatically with `esm3di mlm --push-to-hub true --hub-repo-id org/name`.

### Python API

```python
from esm3di.predict import predict_records
records = [("prot1", "MKTAYIAKQRQISFVKSHFSRQ")]
for header, three_di, confidence in predict_records(records, checkpoint="runs/3di_small/best.pt"):
    print(header, three_di)

from esm3di.checkpoints import load_3di_checkpoint      # (ESM3DiModel, ckpt dict)
from esm3di.fastas2foldseekdb import create_foldseek_db_from_fastas
```

`from esm3di import ESM3DiModel, read_fasta` and `ESM3DiModel.predict_from_fasta(...)`, which the Colab notebook uses, still work.

## Repository layout

| Path | Contents |
|---|---|
| [esm3di/cli.py](esm3di/cli.py) | `esm3di` command: `prepare`, `mlm`, `train`, `predict`, `foldseekdb`, `pipeline` |
| [esm3di/prepare_data.py](esm3di/prepare_data.py) | structures → paired AA/3Di/pLDDT FASTAs via FoldSeek |
| [esm3di/train_mlm.py](esm3di/train_mlm.py) | stage 1: ESM++ MLM fine-tuning |
| [esm3di/train_3di.py](esm3di/train_3di.py) | stage 2: 3Di token classification |
| [esm3di/checkpoints.py](esm3di/checkpoints.py) | checkpoint loading, MLM → 3Di backbone transfer, HF revision pinning |
| [esm3di/ESM3di_model.py](esm3di/ESM3di_model.py) | `ESM3DiModel` (backbone + LoRA + heads), dataset utilities |
| [esm3di/predict.py](esm3di/predict.py), [esm3di/foldseek_db.py](esm3di/foldseek_db.py) | inference and FoldSeek DB writer |
| [esm3di/experimental/](esm3di/experimental/) | unsupported earlier approaches and data tools |
| [configs/](configs/) | recipes for the supported pipeline |
| [analyses/](analyses/) | benchmarks, case studies and scratch material (not part of the package) |
| [notebooks/](notebooks/), [reports/](reports/) | notebooks (incl. Colab inference) and training reports |

## Citation

If you use this code, please cite:

- ESM++: [Synthyra, 2024](https://huggingface.co/Synthyra)
- ESM-2: Lin et al., 2022
- LoRA: Hu et al., 2021
- FoldSeek / 3Di: van Kempen et al., 2023

## License

MIT License

## Funding

Funded by NIH through the [Pathogen Data Network](https://www.pathogendatanetwork.org).

*This resource is supported by the National Institute Of Allergy And Infectious Diseases of the National Institutes of Health under Award Number U24AI183840. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.*

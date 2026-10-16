# ESM3Di

![ESM3Di logo: viral protLLM](https://raw.githubusercontent.com/DessimozLab/ESM3di/master/logoEMS3Di.png)

Predict per-residue **3Di structural tokens** (the FoldSeek structural alphabet) directly from amino-acid sequences, so you can build FoldSeek databases and run structure-aware searches for proteins without structures.

**Start here:** [install](#installation), then
[predict 3Di or build a FoldSeek database](#quick-start-fasta-to-3di-or-a-foldseek-database)
using the [pretrained Hugging Face weights](https://huggingface.co/cactuskid13/ESM3di_Small_MLM_3di).

## Scope

ESM3Di is an experimental, dataset-focused effort. Its models and evaluations are intended for viral protein collections and other explicitly curated, specialized structure sets (for example, the BFVD-derived training splits used in this repository).

It is **not** intended to be a general-purpose replacement for ProstT5, nor does this repository claim broad performance equivalence across arbitrary proteins, taxa, or structural datasets. For general 3Di prediction, use ProstT5 or another broadly validated method; use ESM3Di when working within a supported viral or specialized-structure setting and validate performance on the target dataset.

Most users only need an amino-acid FASTA and the pretrained
[ESM3Di small weights on Hugging Face](https://huggingface.co/cactuskid13/ESM3di_Small_MLM_3di):
no structures or training are needed to predict 3Di or build a FoldSeek database.

## Installation

```bash
# Inference only (from this checkout)
python -m venv venv && source venv/bin/activate
pip install -e .

# Full install: inference plus MLM / 3Di training
pip install -e . -e ./training

# Alternatively, Conda with external bioinformatics tools
conda env create -f environment_inference.yml   # inference only
conda env create -f environment.yml             # full training + inference
conda env create -f environment_blackwell.yml   # full, Blackwell GPUs
# Activate the one you created:
conda activate esm3di_inference                 # or esm3di / esm3di_blackwell
```

There are two distributions: **`esm3di`** contains only inference and the
shared ESM++ checkpoint runtime; **`esm3di-training`** adds the trainers,
data preparation, publishing utilities, and training dependencies.
After both distributions are published, use `pip install esm3di` or
`pip install 'esm3di[training]'`. Until then, use the local commands above,
or install both locally built wheels together. See [training installation](training/README.md).
Inference does not require pandas, scipy, scikit-learn, TensorBoard,
SentencePiece, or any experimental architecture implementation.

Maintainers: see the [release and Trusted Publishing guide](.github/PUBLISHING.md).
Automated uploads remain disabled pending confirmation of the bundled Python
MAD script's redistribution license.

All three Conda specs install stable PyTorch 2.9.1 with CUDA 12.8 via pip, plus
FoldSeek, FoldMason, and FastTree via Bioconda. They require a CUDA
12.8-compatible NVIDIA driver for GPU use (Linux driver 570 or newer
recommended); CPU inference also works. The Blackwell spec uses Python 3.11
and includes a PyTorch build with `sm_120` support, rather than an old nightly.
Older CUDA 11.8/PyTorch 2.5 environments cannot load the default Hub model's
pinned remote code.

[FoldSeek](https://github.com/steineggerlab/foldseek) is required for searching
the resulting databases and extracting training labels from structures.
`esm3di-infer foldseek-db` writes the database directly without calling the
FoldSeek executable. For training preparation and the legacy database wrapper,
put FoldSeek on `PATH`, or pass `--foldseek-bin /path/to/foldseek`
(or set `FOLDSEEK_BIN`).

> Keep `transformers` (`>=4.45,<4.49`) and `peft` (`>=0.12,<0.14`) in the ranges given in [requirements.txt](requirements.txt). ESM++ is loaded with `trust_remote_code`, and ESM3Di pins the remote-code revision that works with these versions automatically. Override it with `--hf-revision` if you need to.

After installation, use `esm3di` (or `esm3di-infer`) for inference.
The full install additionally provides `esm3di-learn` for training.
Their module equivalents are `python -m esm3di.inference` and
`python -m esm3di_training`; `python -m esm3di.training` remains a full-install alias.

## Quick start: FASTA to 3Di or a FoldSeek database

The CLI automatically downloads and caches the pretrained adapter and CNN
weights from
[cactuskid13/ESM3di_Small_MLM_3di](https://huggingface.co/cactuskid13/ESM3di_Small_MLM_3di),
and loads their ESM++ backbone. This is also the default when `--model-ckpt`
is omitted. Pass the **Hub repository ID**, not the browser URL.

```bash
# Predict per-residue 3Di tokens
esm3di-infer predict \
  --model-ckpt cactuskid13/ESM3di_Small_MLM_3di \
  --input-fasta proteins.fasta \
  --output-fasta outputs/proteins_3di.fasta

# Predict and build a FoldSeek database in one command
esm3di-infer foldseek-db \
  --model-ckpt cactuskid13/ESM3di_Small_MLM_3di \
  --input-fasta proteins.fasta \
  --output-db dbs/proteins

# Search using another FoldSeek database as the query
foldseek easy-search query_db dbs/proteins hits.m8 tmp --exhaustive-search 1
```

The model uses available GPUs, or CPU when CUDA is unavailable. Use
`--num-gpus 1` to restrict inference to one GPU and `--batch-size 8` to set
the batch size per device. Database output is a bundle of files sharing the
chosen prefix; keep the bundle together.

> **Custom training checkpoints:** replace the Hub ID with your supervised
> checkpoint, for example `--model-ckpt runs/my_model/3di/best.pt`, in either
> command above. The loader reconstructs the saved architecture and label
> vocabulary. An MLM-only directory is not a trained 3Di predictor.

The release inference path truncates sequences to its model window. For long
sequences with a custom `.pt` checkpoint, `esm3di-predict` supports
overlapping-window prediction (see below), including in inference-only installs.

### Other interfaces

- `esm3di-infer` also offers `perplexity` and `foldtree`. Install
  `pip install '.[workflow]'` for `foldtree`; its external tools are managed
  by the bundled Snakemake environments.
  MAD rooting runs the bundled Python script with the workflow environment's
  interpreter (no native MAD executable). It validates a single rooted Newick
  tree and unchanged tip labels before accepting the output. A custom workflow
  `mad_path` must point to a Python script, not the former native binary.
- `esm3di-learn` provides `prepare`, `mlm`, `train`, and `pipeline`.
- Full installs retain legacy training subcommands through `esm3di`, but
  its default help lists only inference commands. The experimental training
  commands are not installed. `esm3di-infer cut` provides polyprotein cleavage prediction;
  `esm3di-cut` and `python -m esm3di.cutting_cli` remain equivalent standalone
  interfaces.

### Polyprotein cleavage prediction

Use a local ESM3Di training checkpoint for embeddings and the separate trained
cut-site classifiers:

```bash
esm3di-infer cut \
  --input-fasta polyproteins.fasta \
  --esm3di-checkpoint /path/to/embedding_model.pt \
  --ffn-checkpoint experiments/analyses/polyprotein_cut/models/cutsite_ffn_all_data.pt \
  --transformer-checkpoint experiments/analyses/polyprotein_cut/models/cutsite_transformer_all_data.pt \
  --classifier ensemble --cutoff 0.5 \
  --output-prefix outputs/polyproteins
```

This writes residue probabilities (`_residue_predictions.csv`), per-sequence
cut summaries (`_sequence_summary.csv`), and product sequences
(`_cleaved_products.fasta`) by default. Select `--output-format fasta` for
cleaved products only, `--output-format csv` for cut-information CSVs only,
or `--output-format both` (the default) for all three files.
A predicted position cuts **after** that residue.
`--classifier ffn` or `transformer` needs only its corresponding classifier
checkpoint. Use `--device cpu` or `cuda` and `--batch-size` as needed.

The cutter can use local `.pt` files as above, or a separate exported cutter
bundle. The default Hub 3Di adapter/CNN repository by itself is not a cutter.
Use the embedding checkpoint on which the cut-site classifiers were trained;
substituting a newly trained backbone is not a validated equivalent.
The archived heads have limited held-out-group performance, and the default
cutoff is not a guarantee of biological cleavage. See
[cut-site methods and checkpoint provenance](experiments/analyses/polyprotein_cut/README.md)
before interpreting predictions. Existing files with the same output prefix
are overwritten, as in the standalone cutter.

#### Portable cutter bundle for Hugging Face

Export only the cut-site classifier heads, referencing the existing Hub
embedding model rather than uploading its backbone again:

```bash
python -m esm3di.inference.cutting_bundle \
  --esm3di-checkpoint /path/to/original_embedding_model.pt \
  --ffn-checkpoint experiments/analyses/polyprotein_cut/models/cutsite_ffn_all_data.pt \
  --transformer-checkpoint experiments/analyses/polyprotein_cut/models/cutsite_transformer_all_data.pt \
  --embedding-repo cactuskid13/ESM3di_Small_MLM_3di \
  --embedding-revision b32e8d308efd435be015122d7a1ca8f2dc671e86 \
  --output-dir exports/cutter
```

Upload the four files in that directory to **`cutter/`** at the root of
`cactuskid13/ESM3di_Small_MLM_3di`, preserving the existing files directly in
`hf_compatible/`. Then use:

```bash
esm3di-infer cut --cutter-model cactuskid13/ESM3di_Small_MLM_3di \
  --input-fasta polyproteins.fasta --output-prefix outputs/polyproteins
```

This bundle uses a custom ESM3Di loader, not `AutoModel.from_pretrained`.
It contains both heads (~8.6 MB total), a JSON config, and documentation;
weights use safetensors with SHA256 integrity checks. The existing Hub model
and its base-model/tokenizer revision are pinned in the config and reused.
No `embedding.safetensors` is needed for this heads-only bundle.
`--cutter-revision` pins the **bundle's Hub revision**; it does not
change the embedding backbone. `--cutter-subfolder` defaults to
`cutter`. For local use, pass the exported directory itself to
`--cutter-model`. Do not combine bundles with individual classifier overrides.

The upload and real inference validation of this cutter bundle are pending;
the command above is for use after the bundle is uploaded and verified.

### FoldMason alignment and FastTree inference

Both Conda environment specs install [FoldMason](https://github.com/steineggerlab/foldmason)
and [FastTree](https://github.com/morgannprice/fasttree) from Bioconda.
For an existing environment or a pip installation, install the external tools:

```bash
conda install -c conda-forge -c bioconda foldmason=4.dd3c235 'fasttree>=2.2.0'

# Database to alignment and tree
esm3di-infer foldmason-fasttree --input-db dbs/proteins \
  --output-dir results/proteins_tree --threads 8

# Or run the two stages separately
esm3di-infer foldmason --input-db dbs/proteins \
  --output-dir results/proteins_alignment --threads 8
esm3di-infer fasttree --input-alignment results/proteins_alignment/alignment_aa.fa \
  --output-tree results/proteins.nwk --threads 8
```

FoldMason runs `structuremsa --fast 1 --refine-iters 0`, which supports
predicted-3Di databases without coordinates. FastTreeMP receives the
**amino acid alignment**, not the 3Di alignment, and runs
`-fastest -noml -spr 0 -nosupport`: approximate neighbor joining followed by
minimum-evolution NNI, with no maximum-likelihood refinement or support values.
The resulting tree is unrooted and its branch lengths are distance-based.

Output directories must be new; existing trees and logs are never overwritten.
The combined run retains `alignment_aa.fa`, `alignment_3di.fa`, FoldMason's guide
tree, `tree.nwk`, diagnostic logs, and the FoldMason command in `run.txt`.
Failed runs retain logs; retry with a new output location.
Use `--foldmason-bin` / `FOLDMASON_BIN` and `--fasttree-bin` / `FASTTREE_BIN`
for tools installed elsewhere. These commands do not load model weights or
require Snakemake. See the [original experiment methods](experiments/analyses/fasttree_NJ/README.md)
for interpretation and limitations.

## Training: MLM followed by supervised 3Di

The supported recipe is a two-stage fine-tune of
[ESM++](https://huggingface.co/Synthyra): unsupervised masked-language-model
(MLM) training on domain sequences, followed by supervised 3Di training from
the MLM backbone with LoRA adapters and a CNN head. FoldSeek supplies labels
from structures; pLDDT can weight the loss. Other model approaches are
archived in [experiments/legacy_runtime/](experiments/legacy_runtime) for reference only;
they are excluded from both installable distributions.

### One-command training pipeline

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

To sanity-check predictions from a source checkout, run
`python experiments/legacy_runtime/test_output_diversity.py proteins_3di.fasta`.
It warns if the output collapses to a few tokens.

## Configuration files

Both trainers accept `--config file.json`. Keys are the long option names with underscores (`batch_size`, `init_from`, …), and keys starting with `_` are treated as comments. Settings are applied in this order, each overriding the previous: built-in defaults, then the config file, then `--data-dir` paths, then explicit command-line flags. Relative paths are resolved against the current working directory.

| Config | Purpose |
|---|---|
| [configs/mlm_esmpp_small.json](configs/mlm_esmpp_small.json), [configs/mlm_esmpp_large.json](configs/mlm_esmpp_large.json) | MLM stage on `data/` |
| [configs/3di_esmpp_small_from_mlm.json](configs/3di_esmpp_small_from_mlm.json), [configs/3di_esmpp_large_from_mlm.json](configs/3di_esmpp_large_from_mlm.json) | 3Di stage from the MLM output |
| [configs/bfvd/](configs/bfvd) | Same recipe on the BFVD viral structure splits (absolute paths for our cluster) |

Older experimental options (iterative heads, T5, auxiliary tracks, etc.) are
rejected explicitly. Their historical source is archived, not installed.

## Checkpoints and compatibility

**3Di checkpoint (`.pt`).** A `torch.save` dict with `model_state_dict` (complete model: backbone, LoRA, and head), `label_vocab`, `mask_label_chars`, `args`, `epoch`, `optimizer_state_dict`, and, for new runs, a `provenance` record of which MLM checkpoint the backbone came from. The architecture is rebuilt from `args`, so prediction only needs the `.pt` file.

**MLM checkpoint (directory).** A standard HuggingFace model directory, plus `esm3di_mlm.json` with the training metadata. Directories containing only a PEFT adapter are merged into their base model on load.

**Compatibility with older checkpoints:**

- ESM++ LoRA/CNN checkpoints written by the old `esmretrain.py` / `esmretrain_mlm.py`
  (including `checkpoints/esm3di_from_mlm/*.pt` and the `esmc_viral_ft_large`
  MLM dirs) remain supported. `.pt` inference needs only `esm3di`;
  `esm3di-learn train --init-from` requires the training distribution.
  The `DataParallel` `module.` prefix is handled. The default Hub adapter/CNN
  bundle is not the same format as a training `.pt` checkpoint.
- Checkpoints from older PEFT versions that wrapped extra modules with LoRA are merged into dense weights on load, and their outputs are unchanged.
- With the training distribution installed, `python -m esm3di.esmretrain` and
  `python -m esm3di.esmretrain_mlm` still forward to the supported trainers.
- Retired ESM-2, T5, linear-only, iterative, transformer-head, pLDDT-head, and
  auxiliary-track architectures are no longer reconstructed by the supported
  loader. Their historical source remains in the archive. This does not
  remove the separate, supported transformer cut-site classifier.

> ⚠️ **Note on earlier "MLM → 3Di" runs.** In the previous trainer, the MLM checkpoint given to the 3Di stage was not actually loaded into the backbone, so those 3Di models were trained on the stock ESM++ weights. The new trainer copies the MLM backbone before adding LoRA and logs how many tensors were transferred. Retrain with the new pipeline to get the real benefit of the MLM stage.

### Publishing custom checkpoints

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
| [src/esm3di/inference/](src/esm3di/inference) | inference CLI, predictors, cutter and FoldSeek DB writer |
| [src/esm3di/_core/](src/esm3di/_core) | shared ESM++ LoRA/CNN model, checkpoint reconstruction, MLM transfer and revision pinning |
| [src/esm3di/workflow/](src/esm3di/workflow) | optional Snakemake assets, including platform-independent Python MAD rooting |
| [training/](training) | separately installable `esm3di-training` distribution |
| [training/src/esm3di_training/](training/src/esm3di_training) | supported trainers, structure preparation, losses and publishing |
| [experiments/legacy_runtime/](experiments/legacy_runtime) | non-installable historical architectures and data tools |
| [configs/](configs) | recipes for the supported pipeline |
| [experiments/](experiments/) | biological case studies, monophyly trees, model-development and scratch material (not part of the package) |
| [benchmarks/](benchmarks/) | alignment, fold recognition, random baselines and benchmark datasets (not part of the package) |
| [notebooks/](experiments/notebooks), [reports/](experiments/reports) | notebooks (incl. Colab inference) and training reports |

## Research documentation

Start with the [experiment overview](experiments/README.md),
[experiment index](experiments/analyses/README.md), or
[benchmark overview](benchmarks/README.md). Each guide distinguishes saved
historical results from analyses that need a fresh rerun.

| Topic | Documentation |
|---|---|
| Alignment, fold recognition, and random 3Di baseline | [Benchmark comparisons](benchmarks/comparisons/README.md) |
| Archived benchmark search databases | [Proteome database artifacts](benchmarks/proteomes/README.md) |
| Monophyly scoring and large-tree visualization | [Phylogeny experiments](experiments/analyses/phylogeny/README.md) |
| Flavivirus analyses | [Flavivirus overview](experiments/analyses/flavi_redo/README.md), [envelope and NS5 comparisons](experiments/analyses/flavi_redo/simple_flavi/README.md) |
| Ebola protein analyses | [Ebola experiments](experiments/analyses/ebola_data/README.md) |
| HIV mutation analysis | [HIV experiments](experiments/analyses/HIV_mutations/README.md) |
| Polyprotein cut-site prediction | [Cleavage experiments](experiments/analyses/polyprotein_cut/README.md) |
| Structure-aware alignment and tree construction | [FastTree/NJ experiments](experiments/analyses/fasttree_NJ/README.md) |
| Historical training curves and reports | [Archived curves](experiments/analyses/checkpoints_notebook/README.md), [reports](experiments/reports/README.md) |

Historical `esm3di_from_mlm` checkpoints did not receive the intended MLM
backbone weights. Their saved research results must not be interpreted as
evidence of an MLM fine-tuning benefit; the corrected run uses separate artifacts.

## Citation

If you use this code, please cite:

- ESM++: [Synthyra, 2024](https://huggingface.co/Synthyra)
- LoRA: Hu et al., 2021
- FoldSeek / 3Di: van Kempen et al., 2023

## License

MIT License

## Funding

Funded by NIH through the [Pathogen Data Network](https://www.pathogendatanetwork.org).

*This resource is supported by the National Institute Of Allergy And Infectious Diseases of the National Institutes of Health under Award Number U24AI183840. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.*

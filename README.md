# ESM3Di

![ESM3Di logo: viral protLLM](https://raw.githubusercontent.com/DessimozLab/ESM3di/master/logoEMS3Di.png)

Predict per-residue **3Di structural tokens** from amino-acid sequences, build
FoldSeek search databases, and predict polyprotein cleavage sites.
Pretrained weights are downloaded automatically from
[Hugging Face](https://huggingface.co/cactuskid13/ESM3di_Small_MLM_3di);
you do not need structures or training checkpoints to run predictions.

**Scope:** ESM3Di is an experimental resource for viral proteins and curated,
specialized structure collections. It is not a general-purpose replacement
for ProstT5. Validate predictions on your target dataset.

## Installation

From a checkout of this repository:

```bash
git clone https://github.com/DessimozLab/ESM3di.git
cd ESM3di
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Use Python 3.10–3.12. CPU inference works; compatible NVIDIA GPUs are used
automatically. The first prediction needs internet access to download weights.

For Conda installations including FoldSeek, FoldMason and FastTree:

```bash
conda env create -f environment_inference.yml
conda activate esm3di_inference
```

For training, install `pip install -e . -e ./training`, or use
[environment.yml](environment.yml). Blackwell GPU users can use
[environment_blackwell.yml](environment_blackwell.yml). The Conda environments
use PyTorch with CUDA 12.8; a compatible NVIDIA driver is required
(Linux driver 570 or newer recommended). Keep the installed
`transformers` and `peft` versions within the supplied dependency ranges.

## Predict 3Di or build a FoldSeek database

```bash
# Amino-acid FASTA → predicted 3Di FASTA
esm3di-infer predict \
  --input-fasta proteins.fasta \
  --output-fasta outputs/proteins_3di.fasta

# Amino-acid FASTA → FoldSeek database
esm3di-infer foldseek-db \
  --input-fasta proteins.fasta \
  --output-db dbs/proteins

# Search the resulting database
foldseek easy-search query_db dbs/proteins hits.m8 tmp --exhaustive-search 1
```

The default model is
[`cactuskid13/ESM3di_Small_MLM_3di`](https://huggingface.co/cactuskid13/ESM3di_Small_MLM_3di).
Use `--model-ckpt` to select another compatible Hub repository or your own
supervised ESM++ LoRA/CNN `.pt` checkpoint. An MLM-only directory is not a
trained 3Di predictor.

Use `--num-gpus 1` to restrict inference to one GPU and `--batch-size 8` to set
the batch size per device. Keep all database files sharing the output prefix
together. FoldSeek must be installed for searches, but database writing itself
does not need the FoldSeek executable.

**Long sequences are supported by default, including with Hugging Face models.**
Prediction, confidence output, FoldSeek database generation and cutter embeddings
use overlapping windows and retain every residue. The standard inference window
is up to 1,024 tokens including special tokens (normally 1,022 residues), reduced
if the tokenizer has a smaller limit. Windows overlap by 128 residues (less for
very small contexts); 3Di probabilities or cutter embeddings are averaged in
overlaps. Short sequences use a single window. Windowing preserves length, but
does not provide interactions between residues in distant windows.

**Verified with real model weights:** the default Hugging Face model, a legacy
supervised `.pt` checkpoint and a corrected MLM-to-3Di `.pt` checkpoint each
produced exactly one 3Di token per input residue for sequences of 22, 1,022,
1,023 and 2,051 residues. Predictions matched the existing standalone windowed
predictor, and short-sequence predictions were unchanged. Confidence output and
paired AA/3Di FoldSeek records also retained every residue. The Hugging Face
cutter produced probabilities for all 2,051 residues of a test polyprotein,
and concatenating its predicted products reconstructed the complete input.
These checks verify output completeness and compatibility, not prediction
accuracy or unlimited sequence length; available memory still limits inference.
Existing model weights and checkpoints were not modified.

### Prediction options

| Option | Applies to | Behavior |
|---|---|---|
| `--model-ckpt REPO_OR_PATH` | `predict`, `foldseek-db`, `perplexity` | Select compatible Hub weights, a local adapter/CNN directory, or a supervised `.pt` checkpoint |
| `--batch-size N` | `predict`, `foldseek-db`, `perplexity` | Maximum windows per model batch (default: 8); reduce if GPU memory is limited |
| `--num-gpus N` | `predict`, `foldseek-db` | Number of GPUs (default: all available; CPU when none are available) |
| `--no-sort-by-length` | `predict`, `foldseek-db` | Disable length-sorted batching; output record order is preserved either way |
| `--truncate` | `predict`, `perplexity` only | **Opt in** to discarding residues after the first model window; never enabled by default |
| `-v 0` through `-v 3` | All inference commands | Quiet through informational logging (default: 3) |

Truncated predictions are incomplete and must not be paired with full-length AA
sequences for a FoldSeek database. Accordingly, `foldseek-db` and the cutter do
not offer truncation. To deliberately generate only first-window predictions:

```bash
esm3di-infer predict --input-fasta proteins.fasta \
  --output-fasta outputs/first_window_only.fasta --truncate
```

For custom supervised checkpoints, the standalone predictor also supports
explicit window sizes and overlap, plus per-residue confidence digits:

```bash
esm3di-predict --checkpoint runs/my_model/3di/best.pt \
  -i proteins.fasta -o proteins_3di.fasta \
  --window 1022 --overlap 128 --confidence-output confidence.fasta
```

That interface uses the checkpoint's training sequence length by default and
does not truncate. Use `esm3di-infer perplexity --input-fasta proteins.fasta
--output-tsv confidence.tsv` for full-length entropy-derived confidence from
the default Hub model (lower perplexity means a more concentrated prediction).

See the [database wrapper guide](fastas2foldseekdb_README.md) for working with
precomputed AA/3Di FASTAs.

### Other interfaces

- **Polyprotein cutter:** `esm3di-infer cut` downloads the matching Hugging Face
  weights and writes cleaved-product FASTA and cut-information CSVs.
  See [cleavage prediction](#polyprotein-cleavage-prediction) below.
- **Alignment and trees:** `esm3di-infer foldmason-fasttree` builds an alignment
  and an approximate tree from a FoldSeek database.
- **Prediction confidence:** `esm3di-infer perplexity --help`.
- **Training:** `esm3di-learn` provides `prepare`, `mlm`, `train` and `pipeline`.
- **FoldTree workflow:** `esm3di-infer foldtree --help`; requires
  `pip install -e '.[workflow]'` and Conda for its external tools.

## Polyprotein cleavage prediction

```bash
esm3di-infer cut \
  --cutter-model cactuskid13/ESM3di_Small_MLM_3di \
  --input-fasta polyproteins.fasta \
  --output-prefix outputs/polyproteins
```

This downloads both cut-site classifiers and their matching embedding model
from Hugging Face. **No local `.pt` file is required.** Downloaded weights are
cached for subsequent runs.
Long polyproteins are embedded with overlapping windows without truncation;
the cut-site classifiers then process the complete embedding sequence.

By default, the command writes:

| Output | Contents |
|---|---|
| `polyproteins_residue_predictions.csv` | Per-residue probabilities and cleavage calls |
| `polyproteins_sequence_summary.csv` | Cut positions and product counts per sequence |
| `polyproteins_cleaved_products.fasta` | Predicted cleaved products |

Use `--output-format fasta`, `csv`, or `both` (default). Positions are
**1-based and cut after the indicated residue**. The default is an
FFN/transformer ensemble with probability cutoff `0.5`; change it with
`--cutoff`, or select one head with `--classifier ffn` or `transformer`.
Use `--device cpu` or `cuda` and `--batch-size` as needed.
Existing files with the same output prefix are overwritten.

For reproducibility, `--cutter-revision` pins the cutter's Hub commit.
The matching embedding model is already pinned in its configuration.
`esm3di-cut` is an equivalent standalone command.

**Interpretation:** these classifiers have limited held-out-group performance.
A predicted cut is not experimental evidence of biological cleavage.
See [cut-site methods and validation](experiments/analyses/polyprotein_cut/README.md).

## Align sequences and infer a tree

FoldMason and FastTree are included in the Conda environments. To install them
separately:

```bash
conda install -c conda-forge -c bioconda foldmason=4.dd3c235 'fasttree>=2.2.0'

esm3di-infer foldmason-fasttree --input-db dbs/proteins \
  --output-dir results/proteins_tree --threads 8
```

The output includes AA and 3Di alignments, a guide tree, `tree.nwk`, and logs.
Use a new output directory for each run; existing results are not overwritten.

FoldMason uses `structuremsa --fast 1 --refine-iters 0`. FastTree uses the
**amino-acid alignment**, with `-fastest -noml -spr 0 -nosupport`: approximate
neighbor joining and minimum-evolution NNI, not maximum-likelihood refinement.
The tree is unrooted and has no support values. See the
[tree-building methods](experiments/analyses/fasttree_NJ/README.md) for limitations.

## Train on your own sequences and structures

Install the training component first:

```bash
pip install -e . -e ./training
```

The supported recipe uses **ESM++**, unsupervised MLM fine-tuning, then
supervised 3Di training with LoRA and a CNN head. FoldSeek must be on `PATH`
to extract labels from structures, or set `FOLDSEEK_BIN`.

```bash
esm3di-learn pipeline \
  --structures structures/ \
  --sequences viral_proteins.fasta \
  --init-from Synthyra/ESMplusplus_small \
  --out-dir runs/my_model
```

This prepares paired training data, trains the MLM, and writes the supervised
model to `runs/my_model/3di/best.pt`.

| Starting material | Pipeline options |
|---|---|
| Structures plus additional sequences | `--structures structures/ --sequences proteins.fasta` |
| Structures only, including MLM | `--structures structures/ --mlm-on-structure-sequences` |
| Structures only, without MLM | `--structures structures/` |
| Previously prepared data | `--data-dir data/` instead of `--structures` |
| Existing MLM checkpoint | `--init-from runs/mlm/best` without `--sequences` |
| Existing supervised checkpoint | `--init-from model.pt` without a new MLM stage |

Use a new output directory for a fresh run. To continue an interrupted pipeline,
repeat the command with `--resume`.

### Run stages separately

```bash
esm3di-learn prepare --structures structures/ --out-dir data --cluster-split

esm3di-learn mlm --input-fasta data/train_aa.fasta extra_sequences.fasta \
  --val-fasta data/val_aa.fasta --model-name Synthyra/ESMplusplus_small \
  --output-dir runs/mlm --epochs 3 --batch-size 4

esm3di-learn train --data-dir data --init-from runs/mlm/best \
  --out-dir runs/3di --epochs 5 --batch-size 4
```

Preparation accepts PDB/mmCIF files, directories, and gzip-compressed structures.
It produces paired `{all,train,val,test}_{aa,3di,plddt_bins}.fasta` files and a
manifest. `--cluster-split` keeps FoldSeek clusters in one split.
Use `--no-plddt` for experimental structures whose B-factors are not pLDDT.
Custom paired FASTAs must match in header order and residue length.

When pLDDT bins are available, keep labels unmasked and use confidence-weighted
loss. Otherwise, `X` can mark ignored labels. Training defaults use LoRA rank
16, a two-layer CNN, AdamW and mixed precision.

Both trainers accept JSON configs; explicit CLI flags override config values.
Recipes are in [configs/](configs), including
[small MLM](configs/mlm_esmpp_small.json) and
[small supervised](configs/3di_esmpp_small_from_mlm.json).
Use `esm3di-learn mlm --help` or `esm3di-learn train --help` for options.
Individual stages support `--resume auto`; supervised `--init-from model.pt`
starts a new fine-tune rather than restoring optimizer state.

**Evidence:** MLM initialization is supported, but is not guaranteed to improve
your results. The [paired MetaVR comparison](benchmarks/quality/README.md#mlm-versus-3di-only-fine-tuning--3di_alignment_mlm_vs_3di_onlyipynb)
did not show an aggregate improvement for the evaluated epoch-5 checkpoints.
Historical checkpoints named `esm3di_from_mlm` did not receive MLM weights;
do not interpret their results as evidence of an MLM benefit.

## Research and benchmarks

- [Benchmarks](benchmarks/README.md): 3Di accuracy on MetaVR and SCOPe40,
  speed against ProstT5 and ColabFold, MLM versus 3Di-only fine-tuning, and
  the brief communication figures.
- [Phylogeny experiments](experiments/analyses/phylogeny/README.md): monophyly
  scoring and tree visualization.
- [Flavivirus](experiments/analyses/flavi_redo/README.md),
  [Ebola](experiments/analyses/ebola_data/README.md), and
  [HIV](experiments/analyses/HIV_mutations/README.md) analyses.
- [Experiment overview](experiments/README.md) and
  [training reports](experiments/reports/README.md).

## Citation

If you use this code, please cite the underlying methods:

- ESM++: [Synthyra](https://huggingface.co/Synthyra)
- LoRA: Hu et al., 2021
- FoldSeek / 3Di: van Kempen et al., 2023

## License

MIT License. Third-party components retain their own licensing requirements.

## Funding

Funded by NIH through the [Pathogen Data Network](https://www.pathogendatanetwork.org).

*This resource is supported by the National Institute Of Allergy And Infectious Diseases of the National Institutes of Health under Award Number U24AI183840. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.*

# Viral polyprotein cleavage-site comparison

Does the 3Di fine-tuning in ESM3Di produce per-residue representations that transfer better
to a viral task the model was not trained on? Cleavage-site prediction in *Amarillovirales*
polyproteins is the test. The two arms differ only in the embeddings — same sequences,
labels, splits, classifier, hyperparameters and seeds — so this is a representation
comparison, not a claim to be a competitive cleavage predictor.

It is also not a description of the released cutting model, which is trained on all four
genera rather than evaluated leave-one-genus-out.

## Contents

| | |
|---|---|
| `cutsite_esm3di_vs_esmpp_standardised.ipynb` | **Source of Figure 2d.** Leave-one-genus-out, 10 seeds, features z-scored with training-fold statistics. |
| `cutsite_esm3di_vs_esmpp_raw.ipynb` | The same run without standardisation, i.e. the preprocessing the shipped `esm3di cutting` pipeline uses. Sensitivity check. |
| `cut_flavi.ipynb` | The original single-seed exploration this analysis grew out of. Kept for provenance; its results are in `results/esm3di_vs_esmpp/`. |
| `scripts/run_comparison.py` | The comparison itself. Both notebooks are thin drivers over it, so they cannot drift from the command line. |
| `scripts/make_paper_figures.py` | Figure 2d, the alternative panels and the supplementary tables. |
| `scripts/cross_genus_identity.py` | Smith-Waterman identity for all 2,704 cross-genus pairs (Methods). |
| `train_all_data.py` | Retrains the cutting heads on the curated 88-record set (see below). |
| `dataset/`, `models/` | ICTV flat files and alignments; the all-data cutting checkpoints. |

Results directories are named after the protocol that produced them:
`results/esm3di_vs_esmpp_standardised/` (Figure 2d), `results/esm3di_vs_esmpp_raw/`, and
`results/esm3di_vs_esmpp/` (the original single-seed run behind `cut_flavi.ipynb`).
`paper/figure2.ipynb` reads `esm3di_vs_esmpp_standardised/roc_curves_by_genus.png` through
its `PANEL_D_IMAGE` constant.

## Experiment

- **Embeddings:** ESM3Di as the published PEFT adapter from `cactuskid13/ESM3di_Small_MLM_3di`
  (subfolder `hf_compatible`), against stock `Synthyra/ESMplusplus_small` at revision
  `46c5f7d`. Both arms use identical 1024-residue windows with 128-residue overlap, averaged
  over the overlaps, which is what `ESM3DiPredictor` does; the adapter was fine-tuned at
  `max_seq_length: 1024`, and these polyproteins are 2,748–4,023 residues.
- **Labels and sequences:** the 86 ICTV records carrying parsed cleavage annotations — 287,411
  residues, 800 sites, a 0.278% positive rate. Cut positions follow the original rule
  (`FT SITE` annotated `/note="Cleavage"`, first residue of each range). Records without
  annotation are excluded rather than treated as all-negative.
- **Split:** leave one genus out across *Hepacivirus*, *Orthoflavivirus*, *Pegivirus* and
  *Pestivirus*, over whole sequences. Hyperparameters are fixed in advance; no held-out genus
  is used for model selection.
- **Head:** FFN 256→128, dropout 0.45/0.10 with 0.10 input dropout, AdamW lr 1e-3, weight
  decay 3e-4, batch 2048, 12 epochs, plus an OSCAR penalty on the final layer. Ten seeds per
  genus × arm, so 80 fits per run. The seed fixes head initialisation, dropout masks and
  batch order only.
- **Metrics:** ROC AUC, strict residue-level AP, AP after expanding labels by ±2 and ±5
  residues, and macro per-protein top-10 site recovery with non-maximum suppression and
  one-to-one matching within ±5 residues.

Read the tolerance-window APs against their own chance level: dilating by ±5 multiplies the
positives elevenfold, so chance AP rises from 0.003 to about 0.03. A higher AP ±5 than
strict AP is not evidence that predictions land within a few residues.

## Data audit

The parser now also reads single-residue `SITE` features, not only ranged ones, which raises the
count from 800 to 803 positive sites. Results generated before that fix contain 800 sites.

## Curated all-data retraining

`train_all_data.py` trains fresh FFN and Transformer heads from the current
ESM3Di checkpoint on the 86 ICTV records plus the two additional Pestivirus
entries with explicit `SITE /note="Cleavage"` annotations. This is a new,
curated 88-record baseline, not a reconstruction of the released 231-record
training set; the original training records and recipe are unavailable.

The other 35 additional Hepacivirus and Orthoflavivirus entries have no
explicit cleavage `SITE` annotation and are excluded rather than assigned
negative labels or inferred from `CHAIN` boundaries. The two added Pestivirus
sites are at residue 497; one has no evidence qualifier and one is supported
by sequence similarity (`ECO:0000250`). The training manifest records source
file hashes, per-sequence hashes, labels, exclusions, and model settings.
Training outputs go to a new directory and do not overwrite the released
heads.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /home/dmoi/.venvs/esm3di-blackwell/bin/python \
  experiments/analyses/polyprotein_cut/train_all_data.py --dry-run

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /home/dmoi/.venvs/esm3di-blackwell/bin/python \
  experiments/analyses/polyprotein_cut/train_all_data.py
```

By default, the retrained checkpoints and manifest are written to
`models/ictv_88_seed17/`. The command refuses to overwrite an existing output
directory; pass a new `--output-dir` to create another run.

## Running

Both notebooks run top to bottom and rebuild everything, including the embeddings; nothing
is read back from a previous run. A GPU is needed (~20 s per arm for the embeddings, then
about 1 s per fit). Large intermediates go to scratch: `oof_ffn_predictions.csv` is ~400 MB
and never enters the repository.

From the command line, the same two protocols are:

```bash
RUN=/scratch/$USER/cutsite_standardised
ADAPTER=$(python -c "from huggingface_hub import snapshot_download; from pathlib import Path
print(Path(snapshot_download(repo_id='cactuskid13/ESM3di_Small_MLM_3di',
      allow_patterns=['hf_compatible','hf_compatible/**']))/'hf_compatible')")

# Figure 2d
python scripts/run_comparison.py --adapter "$ADAPTER" --seeds 1,2,3,4,5,6,7,8,9,10 \
  --architectures ffn --epochs 12 --fast-loader --standardise \
  --cache-dir /scratch/$USER/cutsite_cache --output-dir "$RUN"

# the raw-feature sensitivity check: drop --standardise

python scripts/make_paper_figures.py \
  --results-dir results/esm3di_vs_esmpp_standardised \
  --predictions "$RUN"/oof_ffn_predictions.csv
```

Then copy `seed_metrics.csv`, `summary.csv` and `run_metadata.json` from `$RUN` into the
results directory; everything else there is a few kB.

### Options added for these runs

All default to the original behaviour, so an existing command line is unaffected.

| flag | effect |
|---|---|
| `--seeds`, `--architectures`, `--epochs` | as named |
| `--adapter` | build the ESM3Di arm from a Hugging Face PEFT adapter instead of a `.pt` checkpoint |
| `--standardise` | z-score features per dimension using training-fold statistics (ffn head only) |
| `--fast-loader` | keep the training fold on the GPU instead of streaming it through a `DataLoader`; about 25x faster and trains the same model, but the batch order comes from the device RNG, so runs are statistically equivalent rather than bit-identical |
| `--full-length-embeddings` | one forward pass per sequence instead of 1024-residue windows. Extrapolates well past the adapter's training length and is not reachable through the shipped API; a sensitivity check only |

`run_metadata.json` records which of these were used, along with the checkpoint hash, seed
list, device and metric definitions, so a results directory describes its own protocol.

### Fold divergence (Methods)

```bash
python scripts/cross_genus_identity.py
```

Aligns all 2,704 cross-genus pairs (Smith-Waterman, BLOSUM62). No pair exceeds ~32% identity
within its aligned region, and for every genus pair except the sister genera
*Hepacivirus*–*Pegivirus* the alignment covers under 7% of the shorter sequence — so a
held-out genus has no training-set homologue detectable above the twilight zone.

## Data audit

The original notebook evaluated 231 alignment rows, including repeated protein identifiers
and proteins with no parsed cleavage annotation, the latter converted to all-negative masks.
This comparison uses the 86 unique ICTV records with parsed labels instead, so its numbers
are not directly comparable with the historical notebook results.

A caution for anyone reusing `oof_ffn_predictions.csv`: `run_comparison.py` appends to it, so
an interrupted and resumed run can write a `(genus, arm, seed, sequence)` block twice.
Duplicates are not harmless — the pick-distance analysis works in per-sequence row-index
space, so a doubled sequence doubles every distance. `make_paper_figures.py` now drops
duplicates and prints a warning when it finds any.

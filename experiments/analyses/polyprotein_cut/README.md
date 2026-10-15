# Viral polyprotein cleavage-site analysis

`cut_flavi.ipynb` is the archive's cut-site prediction analysis. It explores
whether frozen per-residue ESM3Di embeddings can support cleavage-site
prediction in annotated viral polyproteins, and whether the classifiers
generalize to a held-out viral group. This is supervised residue-level
prediction from annotated cleavage labels; it is **not** the all-vs-all
polyprotein-region clipping/search analysis elsewhere in the archive. The
notebook reads FASTA and alignment files under
`dataset/Amarillovirales/<group>/`, parses ICTV-style annotation files for
cleavage-site ranges, extracts ESM3Di embeddings, and trains two residue-level
binary classifiers: a feed-forward network (FFN) and a sequence-aware
Transformer.

## Data, method, and saved artifacts

The notebook's recorded data-loading outputs show four groups
(Hepacivirus, Orthoflavivirus, Pegivirus, and Pestivirus), 1,748 FASTA
sequences, 231 alignment sequences, and four parsed ICTV files. The labeled
classifier dataset contains 231 sequences, 775,319 residue positions, and
940 positive cut-site labels. Cleavage positions are parsed from annotated
site ranges and converted to residue-level binary targets; positive labels
are therefore highly imbalanced.

The notebook first reports a conventional validation analysis, then performs
group holdout on Orthoflavivirus-CHH (70 training sequences from the other
groups, 161 held-out sequences), sweeps regularization settings, and retrains
the selected heads on all labeled sequences. Its saved outputs report:

- Initial validation ROC AUC / average precision: FFN 0.9926 / 0.2166;
  Transformer 0.9881 / 0.2006.
- Final Orthoflavivirus-CHH holdout ROC AUC / average precision: FFN
  0.9174 / 0.0669; Transformer 0.9139 / 0.0329.

These are notebook-recorded results, not independently reproduced here. The
very low held-out average precision is important in the context of the rare
positive labels. The FFN's initial validation split takes the first 20% of
flattened residue positions without shuffling, while the Transformer uses a
sequence-level split; those initial validation scores are not directly
comparable and are not a substitute for the held-out-group evaluation.

The notebook stores `models/cutsite_ffn_all_data.pt` and
`models/cutsite_transformer_all_data.pt`. It also includes
`models/cutsite_model_bundle_metadata.json` and three inference smoke-test
artifacts: cleaved-product FASTA, residue-level prediction CSV, and
sequence-summary CSV. The metadata records a default cutoff of 0.5 and a
suggested CLI (`python -m esm3di.cutting_cli` or `esm3di-cut`). These are
inference examples, not external validation results.

The same cutter is now available as `esm3di-infer cut`, with the same
arguments as `esm3di-cut` and `python -m esm3di.cutting_cli`. See the
[usage example](../../../README.md#polyprotein-cleavage-prediction).
This integrates the interface only; it does not replace the embedding
checkpoint or retrain the classifiers.

## Checkpoint provenance

The notebook's recorded run extracts embeddings using
`Synthyra/ESMplusplus_small` (revision `46c5f7d`) and loads
`/home/dmoi/projects/ESM3di/checkpoints/esm3di_from_mlm/epoch_5.pt`; the model
bundle metadata records that same absolute path. The referenced checkpoint is
not included in this directory, and the absolute path will not resolve on
another machine. More importantly, the historical `esm3di_from_mlm` training
run did **not** actually load the MLM backbone weights despite its name. These
classifier results must not be described as evidence that MLM fine-tuning
improved cleavage-site prediction. Reproduction requires the exact 3Di
checkpoint (and its compatible base model/revision), in addition to the two
head checkpoints.

## Reproduction constraints and caveats

- The notebook uses a hard-coded former checkout path
  (`/home/dmoi/projects/ESM3di/polyprotein_cut/dataset/Amarillovirales`).
  Update it to this directory's `dataset/Amarillovirales` before running.
- Embedding extraction requires the ESM3Di Python package, the checkpoint's
  compatible Hugging Face model/tokenizer, PyTorch, and (for the recorded run)
  CUDA. The notebook does not fully pin the environment or external model
  files.
- The held-out group is one viral group in this dataset, not a broad
  independent viral-family test. The final classifier checkpoints were
  retrained using all 231 labeled sequences, so they are not the held-out
  models used to report the group-holdout metrics.
- The metadata's ESM3Di checkpoint location is machine-specific. Preserve
  checkpoint hashes, base-model revision, label parsing rules, split, and
  classifier settings when attempting a reproducible rerun.

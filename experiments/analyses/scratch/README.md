# Scratch analysis and compatibility artifacts

This is a working area of historical analysis material and copied/generated artifacts, not a supported module or reproducible training recipe. Its contents are supporting training-history/compatibility material or miscellaneous scratch artifacts, outside the wider archive's featured themes: one flavivirus envelope-protein tree, alignment/fold-recognition benchmarks, Ebola trees, cut-site prediction, and HIV mutation analysis. Paths and provenance are incomplete in several items. Do not infer model performance or MLM-transfer behavior from these files without independent run metadata.

## Loss-curve reconstruction notebook

[`output/jupyter-notebook/mlm-and-residue-loss-curves.ipynb`](output/jupyter-notebook/mlm-and-residue-loss-curves.ipynb) extracts epoch summaries and batch-running averages from a small ESM++ MLM text log and a 3Di classifier TensorBoard event file, then plots the four loss panels used by the [training report](../../reports/training_validation_loss_report.md). It also records the historical BFVD split counts and classifier settings. It reads existing artifacts rather than recalculating losses. The notebook and report both carry an essential caveat: the legacy `esm3di_from_mlm` classifier did **not** load the MLM checkpoint weights. Its curves cannot be interpreted as the outcome of the supported MLM-backbone transfer; see the report for details. The run files and absolute data paths mentioned there are not included here.

## Remote-code snapshots

`esmc_viral_ft_remote_code/final/modeling_esm_plusplus.py` and `esmc_viral_ft_remote_code/hf_compatible/modeling_esm_plusplus.py` are two checked-in snapshots of remote model code. They are identical in this checkout. They include model/data helper and embedding code plus runtime/performance settings; they are retained as scratch compatibility artifacts, not the project's canonical ESM++ implementation or an independently documented training result. Their filename alone does not establish the exact upstream revision or runtime they were intended to support.

## Other scratch files

- `tmp/14412606171491368149/download.sh` is a generated-style FoldSeek database download helper for several named reference databases. It performs network downloads and is not part of the supported prediction/training commands; review its targets and destination assumptions before any use.
- `tmp.csv` contains only a single column with two numeric entries and has no accompanying schema or provenance, so it cannot be interpreted as a dataset or result table from the available context.

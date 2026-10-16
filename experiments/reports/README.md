# Experimental reports

## Featured themes and scope

The wider archive's featured analyses concern one flavivirus envelope-protein tree, alignment and fold-recognition benchmarks, Ebola trees, cut-site prediction, and HIV mutation analysis. This report is supporting training-history documentation, not primary evidence for any of those biological analyses or benchmarks.

## `training_validation_loss_report.md`

Summarizes recorded losses for a three-epoch small ESM++ MLM run and a five-epoch per-residue 3Di classifier run, including BFVD split counts, selected training settings, and a figure reconstructed from the recorded logs and TensorBoard scalars. The report is about those recorded runs; losses from the MLM and classifier objectives are not directly comparable, and the held-out test split is not part of the plotted classifier validation history.

**Important checkpoint-provenance caveat:** although the historical classifier was described as initialized from `esmc_viral_ft/epoch_3/`, the legacy training path did not actually transfer those MLM weights. The resulting `checkpoints/esm3di_from_mlm/*.pt` model therefore represents stock ESM++ plus its 3Di adaptation/head, not an MLM→3Di run. Do not cite its losses or comparisons as evidence for benefit from MLM initialization. The report records this finding and points to the current `--init-from` path, which transfers the backbone before LoRA and logs the transfer count.

The linked figure is `assets/training_validation_loss_curves.png`; the reconstruction notebook is in [`analyses/scratch`](../analyses/scratch/README.md). The report describes data and artifacts at historical local paths, which are not guaranteed to be present in a checkout.

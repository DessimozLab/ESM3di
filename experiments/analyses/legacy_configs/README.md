# Other exploratory configurations

This directory contains historical configuration notes, not a set of ready-to-run configs for the supported training pipeline. These model/loss experiments are outside the archive's featured scientific themes (single flavivirus envelope-protein tree, alignment/fold-recognition benchmarks, Ebola trees, cut-site prediction, and HIV mutation analysis). The files describe distinct experiments/plans; their expectations and recommendations are not measured results. Paths, trainer arguments, and options may be stale. For current supported configuration and precedence, use the repository [README](../../../README.md) and [configs](../../../configs/).

## ESM-2 3B balanced dataset: cyclical focal loss without pLDDT

[`config_esm2_3B_balanced_ADD_PLDDT.txt`](config_esm2_3B_balanced_ADD_PLDDT.txt) is a note on a balanced-dataset ESM-2 3B configuration. It explicitly says the referenced balanced data lack pLDDT-bin FASTAs, so the described current setup uses cyclical focal loss without pLDDT weighting. It outlines how pLDDT weighting could be added if compatible source structures and aligned bins were available. The suggested accuracy gain in the note is an expectation, not a reported measurement.

## ESM++ Large, BFVD full: cyclical focal loss and pLDDT weighting

[`config_esmpp_large_cyclical_bfvd_full_README.txt`](config_esmpp_large_cyclical_bfvd_full_README.txt) documents a proposed ESM++ Large setup for the BFVD full dataset. It specifies pLDDT-weighted cyclical focal loss, a cyclical schedule, Lion optimization, LoRA, and a linear classifier (no CNN head), with a custom epoch size and multi-GPU/mixed-precision settings. The file presents expected training behavior and target confidence-stratified accuracies; neither should be treated as observed outcomes. It is a descriptive text file referring to a JSON config and legacy `esmretrain` invocation, not that JSON config itself.

These historical approaches differ from the supported recipe, which uses ESM++ for MLM fine-tuning followed by supervised 3Di training initialized with the MLM backbone. The old `esm3di_from_mlm` checkpoints did not actually load MLM weights, so their results cannot substantiate the effect of that transfer.

# Other exploratory training scripts

These shell scripts orchestrate earlier, mostly taxon-specific training experiments through the legacy `esm3di.esmretrain` entry point. They are not the supported MLM→3Di pipeline and are outside the archive's featured scientific analyses (single flavivirus envelope-protein tree, alignment/fold-recognition benchmarks, Ebola trees, cut-site prediction, and HIV mutation analysis); in particular, the ProtT5 variant is an exploratory model workflow. The scripts assume their named config/data files and training environment are available at runtime. Their presence does not establish that every named run completed successfully.

## `train_all_classes.sh`

Runs a sequence of ESM++ Small classifier configs for eleven viral class-level datasets, including a Flavi-specific config. It supports `--resume` and looks for the latest epoch checkpoint under each corresponding output directory. It changes to the script's own directory before looking up configs, so those config files are expected alongside the script. It does not perform MLM fine-tuning or establish that its 3Di models started from an MLM-transferred backbone.

## `train_all_orders.sh`

Runs ESM++ Small classifier configs sequentially for ten viral orders, optionally resuming from the latest epoch checkpoint. Config names and output directory names are constructed from each order. This is an exploratory taxon-specific classification batch, not the supported two-stage training flow.

## `train_all_orders_prostt5.sh`

The analogous ten-order orchestration for ProstT5 configs, using the legacy trainer. This is explicitly outside the supported ESM++ recipe and should be treated as experimental.

## `test_prott5_configs.sh`

Checks for two named test FASTAs and parses three named ProtT5 JSON configs, then prints example legacy training commands. It validates JSON syntax and parameter counts only; it does not test model loading, data compatibility, or training. The referenced FASTAs/configs are not part of this scripts directory.

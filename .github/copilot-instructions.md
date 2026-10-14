# ESM3Di Copilot Instructions

## Project Purpose
ESM3Di predicts per-residue 3Di structural tokens from amino-acid sequences and builds FoldSeek databases from the predictions. The supported recipe has two stages, both on **ESM++**:
1. Unsupervised MLM fine-tuning.
2. Supervised 3Di token classification, using LoRA plus a CNN head and starting from the MLM backbone.

Everything else is experimental.

## Read First (Link, Do Not Duplicate)
- Usage, CLI, checkpoint compatibility: [README.md](../README.md)
- FoldSeek DB wrapper: [fastas2foldseekdb_README.md](../fastas2foldseekdb_README.md)
- Recipes: [configs/](../configs)

## Supported Modules
- [esm3di/cli.py](../esm3di/cli.py): the `esm3di` command, with subcommands `prepare`, `mlm`, `train`, `predict`, `foldseekdb`, and `pipeline`.
- [esm3di/prepare_data.py](../esm3di/prepare_data.py): structures → `{all,train,val,test}_{aa,3di,plddt_bins}.fasta` + `manifest.json`, via FoldSeek.
- [esm3di/train_mlm.py](../esm3di/train_mlm.py): stage 1. Writes HF dirs `epoch_N/`, `best/`, `final/`, each with `esm3di_mlm.json`.
- [esm3di/train_3di.py](../esm3di/train_3di.py): stage 2. Writes `epoch_N.pt`, `best.pt`, `last.pt`, `metrics.jsonl`, `train_config.json`.
- [esm3di/checkpoints.py](../esm3di/checkpoints.py): the single place that handles:
  - HF revision pinning;
  - `--init-from` detection (HF id, MLM dir, or 3Di `.pt`);
  - MLM → backbone transfer before LoRA is applied;
  - 3Di checkpoint rebuild, including the legacy-LoRA merge fallback.
- [esm3di/ESM3di_model.py](../esm3di/ESM3di_model.py): `ESM3DiModel` (backbone, LoRA, heads) and dataset helpers.
- [esm3di/predict.py](../esm3di/predict.py), [esm3di/foldseek_db.py](../esm3di/foldseek_db.py): inference (windowed, multi-GPU) and the FoldSeek DB writer.
- [esm3di/esmretrain.py](../esm3di/esmretrain.py), [esm3di/esmretrain_mlm.py](../esm3di/esmretrain_mlm.py): legacy shims. They forward to the new trainers, or to `experimental/*_full.py` when experimental flags are used.
- [esm3di/experimental/](../esm3di/experimental): earlier approaches (ESM-2/T5, iterative heads, tracks, taxon balancing, …). Do not extend these for the main pipeline.

## Hard Conventions
- Paired FASTAs must match by header order and sequence length.
- The 3Di alphabet is 20 tokens (`A-T`). Input is case-insensitive and processed as uppercase. `X` marks masked labels.
- When a pLDDT-bin FASTA is available, keep the 3Di labels unmasked and let the pLDDT-weighted loss handle confidence.
- Config precedence: defaults < config JSON < `--data-dir` files < explicit CLI flags. Keys use underscores; `_`-prefixed keys are comments.
- Prefer `discover_lora_target_modules()` auto-discovery over hardcoded LoRA target lists.
- An MLM backbone must be loaded *before* LoRA is applied, via `ESM3DiModel(backbone_init=...)`. Check the logged transfer report.

## Checkpoints
- A 3Di `.pt` contains at least `model_state_dict`, `label_vocab`, `mask_label_chars`, `args`, `epoch`, `optimizer_state_dict`; new runs also store `provenance`.
- Any change must keep loading working for old checkpoints:
  - `checkpoints/esm3di_from_mlm/*.pt`;
  - `cactuskid13/ESMpp_small_3Di` and `cactuskid13/esm2small_3di`;
  - MLM dirs such as `esmc_viral_ft_large/best`.

## Compatibility And Pitfalls
- Keep `transformers`/`peft` within the ranges in [requirements.txt](../requirements.txt). ESM++ remote code is pinned per transformers version in `checkpoints.resolve_hf_revision`.
- FoldSeek must be on PATH, or set `--foldseek-bin` / `FOLDSEEK_BIN`. `createdb` only treats a list file as a list when its name ends in `.tsv`.
- DataParallel adds a `module.` prefix to state-dict keys. `strip_module_prefix` handles it.
- [requirements.txt](../requirements.txt) and the environment files are the source of truth for dependencies; [pyproject.toml](../pyproject.toml) mirrors them.

## External Dependencies
- FoldSeek.
- HuggingFace `Synthyra/ESMplusplus_{small,large}`. `facebook/esm2_*` is used only for legacy checkpoints.
- Pretrained checkpoints: `cactuskid13/ESMpp_small_3Di` and `cactuskid13/esm2small_3di`.

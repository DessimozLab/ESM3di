# ESM3Di Copilot Instructions

## Project Purpose
ESM3Di trains ESM-2 and ESM++ models with LoRA adapters to predict per-residue 3Di structural tokens from amino acid sequences, then optionally builds FoldSeek databases from predictions.

## Read First (Link, Do Not Duplicate)
- Main usage and installation: [README.md](../README.md)
- FoldSeek DB pipeline details: [fastas2foldseekdb_README.md](../fastas2foldseekdb_README.md)
- Training config examples: [configs/](../configs/)
- Core implementation: [esm3di/ESM3di_model.py](../esm3di/ESM3di_model.py), [esm3di/esmretrain.py](../esm3di/esmretrain.py), [esm3di/fastas2foldseekdb.py](../esm3di/fastas2foldseekdb.py), [esm3di/build_trainingset.py](../esm3di/build_trainingset.py)

## Quick Agent Workflow
1. Pick environment first (usually [environment.yml](../environment.yml); use [environment_blackwell.yml](../environment_blackwell.yml) for Blackwell GPUs).
2. Prefer config-driven training: `python -m esm3di.esmretrain --config configs/<config>.json`.
3. Use module entrypoints for scripts: `python -m esm3di.<module>`.
4. Validate prediction diversity after inference: `python -m esm3di.test_output_diversity <3di_fasta>`.

## Architecture Landmarks
- `ESM3DiModel` in [esm3di/ESM3di_model.py](../esm3di/ESM3di_model.py): model loading, LoRA injection, heads, checkpoint I/O, inference helpers.
- `Seq3DiDataset` in [esm3di/ESM3di_model.py](../esm3di/ESM3di_model.py): AA/3Di pairing, optional pLDDT bins and auxiliary tracks.
- Training loop in [esm3di/esmretrain.py](../esm3di/esmretrain.py): config parsing, losses, mixed precision, DataParallel, checkpointing.
- Inference and FoldSeek DB builder in [esm3di/fastas2foldseekdb.py](../esm3di/fastas2foldseekdb.py): optional multi-GPU subprocess sharding.

## Hard Conventions
- Paired FASTAs must match by header order and sequence length.
- 3Di alphabet is 20 tokens (`a-t`, case-insensitive input, uppercase in processing).
- `X` marks masked low-confidence labels only when using masking-based training.
- If `plddt_bins_fasta` is provided, use original (unmasked) 3Di labels and let weighted loss handle confidence.
- Prefer ESM++ models (`Synthyra/ESMplusplus_small` or `Synthyra/ESMplusplus_large`) for new work unless a task requires ESM-2 parity.
- Prefer `discover_lora_target_modules()` auto-discovery over hardcoded LoRA target lists.
- Config JSON keys use underscores and override CLI arguments.

## Checkpoints
Expect checkpoint dicts to include at least: `model_state_dict`, `label_vocab`, `mask_label_chars`, `args`, `epoch`, `optimizer_state_dict`.

## Compatibility And Pitfalls
- Keep dependency ranges aligned with [requirements.txt](../requirements.txt), especially `transformers` with `peft`.
- FoldSeek must be available in PATH for DB creation workflows.
- Multi-GPU training uses DataParallel; inference load paths may need to handle `module.` prefixes in state dict keys.
- Iterative backbone head workflows may rely on [esm3di/ESM3di_model copy.py](../esm3di/ESM3di_model%20copy.py).
- [setup.py](../setup.py) contains legacy metadata and looser constraints than runtime files; treat [requirements.txt](../requirements.txt) and environment files as source of truth for dependency compatibility.

## External Dependencies
- FoldSeek for 3Di/FoldSeek database generation.
- HuggingFace model hubs for `facebook/esm2_*` and `Synthyra/ESMplusplus_*`.
- Optional pretrained checkpoints: `cactuskid13/esm2small_3di`, `cactuskid13/ESMpp_small_3Di`.

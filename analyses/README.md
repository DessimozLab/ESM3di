# analyses/

Project-specific analyses, benchmarks and scratch material moved out of the repository root during the cleanup. Nothing here is part of the installed `esm3di` package, and the notebooks and scripts may expect data that is not tracked in git. Paths inside them may still reference their old root-level location.

| Folder | Contents |
|---|---|
| `benchmark/`, `benchmark_proteomes/` | Search-sensitivity benchmarks of predicted vs. true 3Di databases |
| `ebola_data/`, `flavi_redo/`, `HIV_mutations/`, `polyprotein_cut/`, `fasttree_NJ/` | Case studies (viral families, mutation scans, polyprotein cleavage, structural trees) |
| `PDN_Data/` | Pathogen Data Network material |
| `checkpoints_notebook/` | Notebook for inspecting checkpoints |
| `scripts/` | Old batch training launchers (`train_all_*.sh`, `test_prott5_configs.sh`). They call `esm3di.esmretrain`, which now forwards to the new trainer |
| `legacy_configs/` | Text notes for old ESM-2 3B / cyclical BFVD runs |
| `scratch/` | Temporary outputs, plus an exported ESM++ remote-code MLM model (`esmc_viral_ft_remote_code/`) |

For the supported training pipeline, see the [top-level README](../README.md).

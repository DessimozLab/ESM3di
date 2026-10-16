# Research archive

This directory holds research material, not the supported runtime package:

- [analyses/](analyses/): historical experiments, datasets, and
  polyprotein cutting classifiers.
- [notebooks/](notebooks/): training, evaluation, visualization, and Colab notebooks.
- [reports/](reports/): historical training reports.

Start with the [analysis index](analyses/README.md), the
[notebook guide](notebooks/README.md), or the
[report guide](reports/README.md) for experiment-specific methods and caveats.
Supporting data and result subdirectories are documented with their parent
experiment rather than treated as independent experiments.
Alignment, fold-recognition, and random-baseline evaluation and benchmark
datasets are filed separately in [benchmarks/](../benchmarks/README.md).

Install the repository in the notebook's environment (`pip install -e .` from
the repository root) before importing `esm3di`. Notebook data and checkpoint
paths are machine-specific and may require adjustment; training checkpoints
remain outside this archive under the repository's `checkpoints/` directory.

The supported training logic is in `esm3di/training/`; inference-release logic
is in `esm3di/inference/`. Historical model approaches remain in the
`esm3di.experimental` package so legacy imports and checkpoints keep working.

## Evidence and model provenance

The retained research themes are alignment and fold-recognition benchmarks,
all monophyly experiments (including flavivirus NS5, envelope-labelled
E/E1/E2 comparisons, and Ebola trees), supervised cut-site prediction, and
HIV mutation analysis. All-versus-all polyprotein-region clipping remains
separate exploratory material, not a featured monophyly result.

Documentation distinguishes implemented methods, saved historical results,
and analyses that require a fresh controlled rerun. The original
`esm3di_from_mlm` 3Di checkpoints did not receive the intended MLM backbone
weights. Do not interpret their filenames or saved comparison figures as
evidence of an MLM pretraining benefit. The corrected run saves separate
artifacts without modifying those historical checkpoints.

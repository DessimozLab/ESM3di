# Benchmark archive

This directory holds historical evaluation material, separate from the
[biological and model-development experiments](../experiments/README.md):

- [comparisons](comparisons/README.md): MetaVR alignment, SCOPe40 fold
  recognition, the frequency-only random 3Di baseline, and supporting taxonomy
  preparation notebooks.
- [proteomes](proteomes/README.md): an archived search-database bundle with
  incomplete construction provenance.

Monophyly scoring and large-tree visualization belong to the
[phylogeny experiments](../experiments/analyses/phylogeny/README.md), not this
benchmark collection.

Install the repository in the notebook environment with `pip install -e .`
from the repository root. Run notebooks from their containing directory;
relative checkpoint paths assume that working directory. External datasets,
tools, and absolute machine-specific paths still need to be supplied. Review
cells that overwrite external outputs before running them.

These are saved research records, not newly reproduced evaluations. Historical
`esm3di_from_mlm` checkpoints did not receive the intended MLM weights, so their
results do not establish an MLM benefit. Current code and supported recipes
remain outside both archives; see the [usage guide](../README.md).

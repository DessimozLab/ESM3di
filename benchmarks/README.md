# Benchmarks

This directory holds the evaluation of ESM3Di against ProstT5 (and structure-based
references), separate from the
[biological and model-development experiments](../experiments/README.md):

| Folder | Content |
|---|---|
| [quality](quality/README.md) | 3Di accuracy against AF3-derived 3Di on MetaVR (bitscore, identity, random baseline), SCOPe40 viral fold recognition, and the MLM-vs-3Di-only checkpoint comparison |
| [speed](speed/README.md) | Runtime and throughput of ESM3Di, ProstT5 and ColabFold vs number of sequences and sequence length |
| [paper](paper/README.md) | Assembled figures for the brief communication (Figure 1 and supplementary figures) |
| [proteomes](proteomes/README.md) | An archived search-database bundle with incomplete construction provenance |

Monophyly scoring, large-tree visualization and the MetaVR/SCOPe40 taxonomy
preparation belong to the
[phylogeny experiments](../experiments/analyses/phylogeny/README.md), not this
benchmark collection.

## Shared plotting

All benchmark and paper figures use the same code, so a method has the same
colour and a panel looks the same wherever it appears:

- [pub_style.py](pub_style.py): matplotlib style (fonts, sizes, Nature column
  widths, method colours) and `save_pub`, which writes every figure as a vector
  PDF and a 600 dpi PNG.
- [fig_panels.py](fig_panels.py): the plotting function for each panel
  (identity/bitscore scatter, identity distributions, SCOPe sensitivity curves,
  runtime curves). Each draws into a given axes from a results table.

Notebooks in `quality/`, `speed/` and `paper/` import both with
`sys.path.insert(0, "..")`, so run them from their own directory.

## Data flow

Each benchmark notebook runs the external tools, writes its per-item results to
`<benchmark>/results/*.csv` and draws its own figures into
`<benchmark>/figures/`. The paper notebooks read only those CSVs, so the paper
figures can be redrawn without rerunning any benchmark:

```
quality/3di_alignment.ipynb  -> quality/results/3di_benchmark.csv    --\
quality/scope_family.ipynb   -> quality/results/scope_sensitivity.csv -+-> paper/*.ipynb -> paper/figure1/, paper/supplementary/
speed/run_benchmark.py       -> speed/results/*.csv                 --/
```

## Running

Install the repository in the notebook environment with `pip install -e .`
from the repository root. Foldseek, ProstT5 weights, LocalColabFold and the
external datasets are not bundled. The notebooks contain absolute,
machine-specific data paths (e.g. `/scratch/fweikert/data`) that must be
adapted before rerunning. Some cells download large datasets or overwrite files
in those data directories; review them before running a notebook top to bottom.

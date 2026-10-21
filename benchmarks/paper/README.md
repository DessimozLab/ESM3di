# Brief communication figures

Notebooks that assemble the paper figures from the benchmark results. They read
only the CSVs in [quality/results](../quality/results) and
[speed/results](../speed/results), and draw every panel with the same functions
as the benchmark notebooks ([fig_panels.py](../fig_panels.py),
[pub_style.py](../pub_style.py)).

| Notebook | Output | Panels |
|---|---|---|
| [figure1.ipynb](figure1.ipynb) | [figure1/](figure1) | a, model architecture; b, 3Di identity ESM3Di vs ProstT5 with random-baseline median; c, SCOPe family-level sensitivity; d, runtime vs number of sequences |
| [supplementary_figures.ipynb](supplementary_figures.ipynb) | [supplementary/](supplementary) | Suppl. Fig. 1: bitscore scatter, identity distributions, SCOPe superfamily and fold sensitivity. Suppl. Fig. 2: ESM3Di runtime end to end vs inference only, runtime vs sequence length |

Each figure is saved as a vector PDF and a 600 dpi PNG. Panel a of Figure 1 is
drawn separately: save it as `figure1/architecture.png` and rerun the notebook;
until then a placeholder is shown.

Run the notebooks from this directory. They only need the repository's
notebook environment (matplotlib, pandas, numpy), not the benchmark tools.

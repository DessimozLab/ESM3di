# Brief communication figures

The notebooks here only assemble: they place figures and read tables that the analyses write, so a paper figure can
be redrawn without rerunning an analysis. Run them from this directory. They need the repository's notebook
environment; `figure2.ipynb` also needs `pypdf`.

| Folder | Content |
|---|---|
| `figures/` | Figure 1 and Figure 2, as PDF (vector, except the raster tree of Figure 2c) and 600 dpi PNG |
| `supplementary/` | Supplementary figures and table, under their final numbers |
| `assets/` | Drawn by hand: the architecture of Figure 1 (`assets/architecture.png`; a placeholder is shown if it is missing) |

## Where each figure comes from

Paths are relative to the repository root. `PHY` is `experiments/analyses/phylogeny`.

| Figure | Notebook here | Source analysis | Source output |
|---|---|---|---|
| Figure 1b, c | `figure1.ipynb` | `benchmarks/quality/3di_alignment.ipynb`, `scope_family.ipynb` | `benchmarks/quality/results/` |
| Figure 1d | `figure1.ipynb` | `benchmarks/speed/plot_results.ipynb` | `benchmarks/speed/results/final2_nseq.csv`, `final2_nseq_esm3di_cli.csv` |
| Figure 2a | `figure2.ipynb` | `PHY/filoviridae_trees.ipynb` | `PHY/figures/filoviridae_gp_species_trees`, `filoviridae_gp_full_trees` |
| Figure 2b | `figure2.ipynb` | `PHY/flaviviridae_trees.ipynb` | `PHY/figures/flaviviridae_e2_trees` |
| Figure 2c | `figure2.ipynb` | `PHY/massive_radial_tree.py` | `PHY/figures/metagenomic/metagenomic_protein_cluster_tree.png` |
| Figure 2d | reserved for the polyprotein cleavage results | | |
| Suppl. Fig. 2 (accuracy) | `supplementary_figures.ipynb` | `benchmarks/quality/` | `benchmarks/quality/results/` |
| Suppl. Fig. 3 | `supplementary_figures.ipynb` | `benchmarks/speed/plot_results.ipynb` | `benchmarks/speed/results/final2_length_bins_plms.csv`, `final_length_bins_esm3di_cli.csv` |
| Suppl. Fig. 4 | copied by `supplementary_figures.ipynb` | `PHY/filoviridae_trees.ipynb` | `PHY/figures/filoviridae_gp_filtered_trees` |
| Suppl. Fig. 5 | copied | `PHY/flaviviridae_trees.ipynb` | `PHY/figures/flaviviridae_glycoprotein_trees` |
| Suppl. Fig. 6 | copied | `PHY/flaviviridae_jackknife.ipynb` | `PHY/figures/flaviviridae_jackknife_difference` |
| Suppl. Fig. 7 | `supplementary_figures.ipynb` | `PHY/massive_radial_tree.py` | `PHY/figures/metagenomic/metagenomic_supplementary.png`, `.svg` |
| Suppl. Table 1 | copied by `supplementary_figures.ipynb` | `PHY/filoviridae_trees.ipynb` | `PHY/results/filoviridae_gp_accessions.csv` |

The NP trees, the GP amino-acid species tree, the amino-acid comparison of the genus relationships, the earlier jackknife figure (`flaviviridae_jackknife`), the pairwise amino-acid identity (`PHY/pairwise_identity.ipynb`) and the
runtime against sequence length with 32 sequences per bin are not part of the brief communication; they remain in `PHY/figures/` and `benchmarks/speed/`. Figure legends are in [figure_legends.md](figure_legends.md).

The tree figures are SVG (editable text), PDF and 600 dpi PNG at their final size in `PHY/figures/`.

## Order of a full rebuild

1. The analyses: `benchmarks/quality/`, `benchmarks/speed/`, and in `PHY/`: `filoviridae_trees`, `flaviviridae_trees`,
   `flaviviridae_jackknife`, `flaviviridae_ultrametricity`, `pairwise_identity` (after `filoviridae_trees`, whose
   accession table it reads).
2. The notebooks here: `figure1`, `figure2`, `supplementary_figures`.

The shared plotting code is in [benchmarks/](../benchmarks/README.md#shared-plotting).

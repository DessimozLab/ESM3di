# Brief communication figures

The notebooks here only assemble: they place figures and read tables that the analyses write, so a paper figure can
be redrawn without rerunning an analysis. Run them from this directory. They need the repository's notebook
environment; `figure2.ipynb` also needs `pypdf`.

| Folder | Content |
|---|---|
| `figures/` | Figure 1 and Figure 2, as PDF (vector, except the raster tree of Figure 2c) and 600 dpi PNG |
| `supplementary/` | Supplementary figures and tables, under their final numbers |
| `assets/` | Drawn by hand: the architecture of Figure 1 (`assets/architecture.png`; a placeholder is shown if it is missing) and the LoRA diagram of Supplementary Fig. 1 (`assets/networkLORA_bottom.svg`) |

## Where each figure comes from

Paths are relative to the repository root. `PHY` is `experiments/analyses/phylogeny`.

| Figure | Notebook here | Source analysis | Source output |
|---|---|---|---|
| Figure 1b, c | `figure1.ipynb` | `benchmarks/quality/3di_alignment.ipynb`, `scope_family.ipynb` | `benchmarks/quality/results/` |
| Figure 1d | `figure1.ipynb` | `benchmarks/speed/plot_results.ipynb` | `benchmarks/speed/results/final2_nseq.csv`, `final2_nseq_esm3di_cli.csv` |
| Figure 2a | `figure2.ipynb` | `PHY/filoviridae/filoviridae_trees.ipynb` | `PHY/filoviridae/figures/filoviridae_gp_species_trees`, `filoviridae_gp_full_trees` |
| Figure 2b | `figure2.ipynb` | `PHY/flaviviridae/flaviviridae_trees.ipynb` | `PHY/flaviviridae/figures/flaviviridae_e2_trees` |
| Figure 2c | `figure2.ipynb` | `PHY/metagenomic_trees/` | `PHY/metagenomic_trees/figures/metagenomic_protein_cluster_tree.png` |
| Figure 2d | `figure2.ipynb` | `experiments/analyses/polyprotein_cut/cutsite_esm3di_vs_esmpp_standardised.ipynb` | `experiments/analyses/polyprotein_cut/results/esm3di_vs_esmpp_standardised/roc_curves_by_genus.png` |
| Suppl. Fig. 1 | drawn by hand | | `paper/assets/networkLORA_bottom.svg` |
| Suppl. Fig. 2 (accuracy) | `supplementary_figures.ipynb` | `benchmarks/quality/` | `benchmarks/quality/results/` |
| Suppl. Fig. 3 | `supplementary_figures.ipynb` | `benchmarks/speed/plot_results.ipynb` | `benchmarks/speed/results/final2_length_bins_plms.csv`, `final_length_bins_esm3di_cli.csv` |
| Suppl. Fig. 4 | copied by `supplementary_figures.ipynb` | `PHY/filoviridae/filoviridae_trees.ipynb` | `PHY/filoviridae/figures/filoviridae_gp_filtered_trees` |
| Suppl. Fig. 5 | copied | `PHY/flaviviridae/flaviviridae_trees.ipynb` | `PHY/flaviviridae/figures/flaviviridae_glycoprotein_trees` |
| Suppl. Fig. 6 | copied | `PHY/flaviviridae/flaviviridae_jackknife.ipynb` | `PHY/flaviviridae/figures/flaviviridae_jackknife_difference` |
| Suppl. Fig. 7 | `supplementary_figures.ipynb` | `PHY/metagenomic_trees/` | `PHY/metagenomic_trees/figures/metagenomic_supplementary.png`, `.svg` |
| Suppl. Table 1 | copied by `supplementary_figures.ipynb` | `PHY/pairwise_identity/pairwise_identity.ipynb` (after `filoviridae_trees`) | `PHY/pairwise_identity/results/pairwise_identity.csv` |
| Suppl. Table 2 | copied by `supplementary_figures.ipynb` | `experiments/analyses/polyprotein_cut/scripts/make_paper_figures.py` | `experiments/analyses/polyprotein_cut/results/esm3di_vs_esmpp_standardised/supp_table_metrics_formatted.csv` |

The *Filoviridae* GP accession list (`PHY/filoviridae/results/filoviridae_gp_accessions.csv`) is no longer a supplementary
table; the Methods point to it on Zenodo. The NP trees, the GP amino-acid species tree, the amino-acid comparison of the genus relationships, the earlier jackknife figure (`flaviviridae_jackknife`), the pairwise identity figure and the
runtime against sequence length with 32 sequences per bin are not part of the brief communication; they remain in the `figures/` folders of `PHY/` and `benchmarks/speed/`. Figure legends are in [figure_legends.md](figure_legends.md).

The tree figures are SVG (editable text), PDF and 600 dpi PNG at their final size in the `figures/` folders of `PHY/`.

## Order of a full rebuild

1. The analyses: `benchmarks/quality/`, `benchmarks/speed/`, and in `PHY/`: `filoviridae_trees`, `flaviviridae_trees`,
   `flaviviridae_jackknife`, `flaviviridae_ultrametricity`, `pairwise_identity` (after `filoviridae_trees`, whose
   accession table it reads), and `experiments/analyses/polyprotein_cut/cutsite_esm3di_vs_esmpp_standardised.ipynb`
   followed by `scripts/make_paper_figures.py` (see its README).
2. The notebooks here: `figure1`, `figure2`, `supplementary_figures`.

The shared plotting code is in [benchmarks/](../benchmarks/README.md#shared-plotting).

# Phylogeny analyses

Structural trees of viral proteins: the analyses behind Figure 2 and its supplementary figures, plus earlier tree
experiments. Separate from [alignment and fold-recognition evaluation](../../../benchmarks/README.md).
[paper/](../../../paper/README.md) only assembles the figures written here, so they keep names that describe their content
rather than their place in the paper.

| Folder | Content |
|---|---|
| [`filoviridae/`](filoviridae/) | *Filoviridae* GP and NP trees (Figure 2a, Supplementary Fig. 4, Supplementary Table 1) |
| [`flaviviridae/`](flaviviridae/) | *Flaviviridae* E, E1 and E2 trees (Figure 2b, Supplementary Figs. 5 and 6), with their input sequences in `inputs/` |
| [`metagenomic_trees/`](metagenomic_trees/README.md) | The large metagenomic trees (Figure 2c, Supplementary Fig. 7): tree building, drawing, inputs |
| [`pairwise_identity/`](pairwise_identity/) | Amino-acid identity across the *Flaviviridae* and *Filoviridae* datasets (not in the paper) |
| [`shared/`](shared/) | `monophyly.py` (scores), `foldtree_resample.py` (rebuilding and jackknifing trees), `tree_plot.py` (tree figures) |
| [`taxonomy_preparation/`](taxonomy_preparation/) | Clade splits of the MetaVR and SCOPe40 FASTAs |
| [`archive/`](archive/) | Earlier exploratory notebooks, kept for the record |

Each analysis folder keeps its notebooks with its own `results/` and `figures/`. The figures and tables of a notebook are
written relative to its folder, so run a notebook from the folder that holds it. The tree runs stay on `/scratch` (not tracked).

## Notebooks of the paper

| Notebook | Contents | Figures |
|---|---|---|
| `filoviridae/filoviridae_trees.ipynb` | GP and NP trees, scores, species clades, jackknife, trees without X-rich sequences, trees of one representative per species with an amino-acid baseline | `filoviridae_gp_species_trees`, `filoviridae_gp_full_trees` (Figure 2a), `filoviridae_gp_filtered_trees` (Supplementary Fig. 4) |
| `filoviridae/filoviridae_jackknife.ipynb` | Paired jackknife of monophyly and ultrametricity, ESM3Di against ProstT5 (GP, NP); no difference is significant and the replicate spread is large, so it is not in the paper. Run with `filoviridae/run_jackknife.sh` (Slurm) | `filoviridae_jackknife_difference` (not in the paper); tables `results/filoviridae_jackknife*.csv` |
| `flaviviridae/flaviviridae_trees.ipynb` | E, E1 and E2 trees (ESM3Di, AlphaFold2), monophyly and ultrametricity, agreement of ESM3Di 3Di with the structures | `flaviviridae_e2_trees` (Figure 2b), `flaviviridae_glycoprotein_trees` (Supplementary Fig. 5) |
| `flaviviridae/flaviviridae_jackknife.ipynb` | Paired jackknife of the monophyly scores and ultrametricity | `flaviviridae_jackknife_difference` (Supplementary Fig. 6); table `results/flaviviridae_jackknife_summary.csv` |
| `flaviviridae/flaviviridae_ultrametricity.ipynb` | Robustness of the ultrametricity of ESM3Di and AlphaFold2 trees (jackknife, root, clades, distance, structure quality) | `flaviviridae_jackknife` (not in the paper) |
| `flaviviridae/flaviviridae_group_auc.ipynb` | Separation of the annotated groups by pairwise scores | `flaviviridae_group_roc` (not in the paper) |
| `pairwise_identity/pairwise_identity.ipynb` | Pairwise amino-acid identity of the sequences behind the trees, from MAFFT alignments; reads the accession table of `filoviridae/` | `pairwise_identity` (not in the paper) |

`flaviviridae/figures/parts/` holds each tree and legend of Figure 2b separately; it is not tracked (`parts/` is in `.gitignore`).

MAFFT is run single-threaded in `pairwise_identity.ipynb`: with several threads it returns a different alignment from
run to run, which moves the identities in the second decimal place.

## Taxonomy preparation

- `taxonomy_preparation/metavr_taxonomy.ipynb` extracts an IMG/VR protein-cluster to UViG/ICTV taxonomy mapping
  (`data/metavr/metavr_taxonomy.tsv`).
- `taxonomy_preparation/taxonomy_split.ipynb` splits the MetaVR and SCOPe40 amino-acid FASTAs by the clades listed in
  `clades.txt`. The SCOPe40 inputs (`scope40_taxonomy.tsv`, `scope40_viral_aa.fa`) are written by the
  [SCOPe benchmark](../../../benchmarks/quality/README.md) and must be copied into `data/scope/` there.

These are supporting data-preparation materials, not independent results. They read and write a local `data/` directory,
which is not tracked.

## Archive

- `archive/real_tree_visualize_monophyly.ipynb` scores flavivirus NS5, mixed E/E1/E2, and Ebola GP, NP, and L trees. A node
  qualifies when all descendant tip labels share the group prefix before the underscore. The reported fraction counts
  qualifying internal nodes, including nested nodes; it is not a fraction of recovered taxonomic groups or a measure of
  phylogenetic accuracy. Saved NS5 scores include 435/460 (ESM3Di) and 437/460 (amino acid/Mifsud). Mixed E/E1/E2 scores
  include 556/625 (ESM3Di) and 490/625 (AlphaFold/FoldTree). Ebola GP ESM3Di/FoldTree fractions are 0.9798/0.9944, NP
  0.9982/0.9985, and L 0.9986/0.9986, as labelled in the notebook.
- `archive/metavr_125k.ipynb` is the first large-tree visualization (124,931 retained protein tips). It holds an older copy of
  the renderer in a `%%writefile` cell and writes it next to itself; the current renderer is in `metagenomic_trees/`. Drawing
  reduction and taxonomic branch colouring are visualization methods, not independent tests of tree accuracy.
- The [flavivirus archive](../flavi_redo/simple_flavi/README.md) documents related inputs and duplicated score reporting.
  Matching envelope filenames and scores do not establish an independent E-only dataset.

## Running and limitations

Install [requirements.txt](requirements.txt). Run notebooks from their own folder, with the repository installed in their
environment. The external trees and metadata paths used by the cells are not bundled here.

Group labels, root placement, taxon sampling, and tree resolution affect node purity. Historical checkpoint names alone do
not establish MLM adaptation: the old `esm3di_from_mlm` runs did not load the intended MLM backbone.

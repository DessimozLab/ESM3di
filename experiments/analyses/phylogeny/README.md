# Monophyly scoring and large-tree visualization

This archive contains tree experiments formerly grouped with search benchmarks.
It is separate from [alignment and fold-recognition evaluation](../../../benchmarks/README.md).

## Recorded analyses

- `real_tree_visualize_monophyly.ipynb` scores flavivirus NS5, mixed E/E1/E2,
  and Ebola GP, NP, and L trees. A node qualifies when all descendant tip
  labels share the group prefix before the underscore. The reported fraction
  counts qualifying internal nodes, including nested nodes; it is not a
  fraction of recovered taxonomic groups or a measure of phylogenetic accuracy.
  Saved NS5 scores include 435/460 (ESM3Di) and 437/460 (amino acid/Mifsud).
  Mixed E/E1/E2 scores include 556/625 (ESM3Di) and 490/625 (AlphaFold/FoldTree).
  Ebola GP ESM3Di/FoldTree fractions are 0.9798/0.9944, NP 0.9982/0.9985,
  and L 0.9986/0.9986, as labelled in the notebook.
- The [flavivirus archive](../flavi_redo/simple_flavi/README.md) documents
  related inputs and duplicated score reporting. Matching envelope filenames
  and scores do not establish an independent E-only dataset.
- `metavr_125k.ipynb` and `massive_radial_tree.py` implement large-tree
  visualization. The saved notebook describes 124,931 retained protein tips.
  Drawing reduction and taxonomic branch colouring are visualization methods,
  not independent tests of tree accuracy.

## Flaviviridae and Filoviridae trees (Figure 2)

Tree-building notebooks of the brief communication; the runs stay on `/scratch` (not tracked). Each notebook writes
its tables to `results/` and its figures to `figures/`, drawn by `tree_plot.py`. [paper/](../../../paper/README.md)
only assembles them, so the figures keep names that describe their content rather than their place in the paper.

| Notebook | Contents | Figures |
|---|---|---|
| `filoviridae_trees.ipynb` | GP and NP trees, scores, species clades, jackknife, trees without X-rich sequences, trees of one representative per species with an amino-acid baseline | `filoviridae_gp_species_trees`, `filoviridae_gp_full_trees` (Figure 2a), `filoviridae_gp_filtered_trees` (a supplementary figure), `filoviridae_trees`, `filoviridae_species_trees`, `filoviridae_species_identity` (not in the paper); `results/filoviridae_genus_relationships.csv`, the support numbers drawn on the species trees of Figure 2a |
| `flaviviridae_trees.ipynb` | E, E1 and E2 trees (ESM3Di, AlphaFold2), monophyly and ultrametricity, agreement of ESM3Di 3Di with the structures | `flaviviridae_e2_trees` (Figure 2b), `flaviviridae_glycoprotein_trees` |
| `flaviviridae_jackknife.ipynb` | Paired jackknife of the monophyly scores and ultrametricity | `flaviviridae_jackknife_difference` (a supplementary figure); table `results/flaviviridae_jackknife_summary.csv` |
| `flaviviridae_ultrametricity.ipynb` | Robustness of the ultrametricity of ESM3Di and AlphaFold2 trees (jackknife, root, clades, distance, structure quality) | `flaviviridae_jackknife` (monophyly and CV; not in the paper) |
| `pairwise_identity.ipynb` | Pairwise amino-acid identity of the sequences behind the trees, from MAFFT alignments: the three Flaviviridae datasets, and the Filoviridae species representatives split by genus | `pairwise_identity` (a supplementary figure); table `results/pairwise_identity.csv` |
| `flaviviridae_group_auc.ipynb` | Separation of the annotated groups by pairwise scores (not part of the figures) | `flaviviridae_group_roc` (not in the paper) |

The metagenomic trees and their legends (from `massive_radial_tree.py`) are in `figures/metagenomic/`; Figure 2c uses the tree coloured by bacterial host class (`metagenomic_protein_cluster_tree.png`), a supplementary figure the tree coloured by protein cluster (`metagenomic_supplementary.png`, `.svg`).
`figures/parts/` holds each tree and legend of Figure 2b separately; it is not tracked (`parts/` is in `.gitignore`).

MAFFT is run single-threaded in `pairwise_identity.ipynb`: with several threads it returns a different alignment from
run to run, which moves the identities in the second decimal place.

## Taxonomy preparation

- `metavr_taxonomy.ipynb` extracts an IMG/VR protein-cluster to UViG/ICTV
  taxonomy mapping (`data/metavr/metavr_taxonomy.tsv`).
- `taxonomy_split.ipynb` splits the MetaVR and SCOPe40 amino-acid FASTAs by the
  clades listed in `clades.txt`. The SCOPe40 inputs (`scope40_taxonomy.tsv`,
  `scope40_viral_aa.fa`) are written by the
  [SCOPe benchmark](../../../benchmarks/quality/README.md) and must be copied
  into `data/scope/` here.

These are supporting data-preparation materials, not independent results.
They moved here from `benchmarks/comparisons/`.

## Running and limitations

Run notebooks from this directory, with the repository installed in their
environment. Supply the external Newick trees and metadata paths used by the
cells; these inputs are not bundled here. The taxonomy-preparation notebooks
read and write the local `data/` directory, which is not tracked.

The standalone renderer needs NumPy and Matplotlib. Run
`python massive_radial_tree.py --help` for input and rendering options.
The visualization notebook contains a `%%writefile` cell that rewrites the
local renderer; review it before execution. No recorded results were rerun
or changed during relocation.

Group labels, root placement, taxon sampling, and tree resolution affect node
purity. Historical checkpoint names alone do not establish MLM adaptation:
the old `esm3di_from_mlm` runs did not load the intended MLM backbone.

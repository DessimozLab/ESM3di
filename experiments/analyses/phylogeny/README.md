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

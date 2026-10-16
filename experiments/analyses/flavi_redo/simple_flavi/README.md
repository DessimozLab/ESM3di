# Flavivirus envelope and NS5 monophyly experiments

## Biological question

The retained analyses include the envelope-labelled E/E1/E2 tree comparisons
and the NS5 polymerase tree comparisons. They ask whether the inferred trees group labelled
flavivirus taxa into monophyletic subtrees. Their monophyly fraction is a
descriptive label-purity score, not a benchmark of phylogenetic accuracy.

## Inputs and method

`nwk/E_esm3di.nwk` is the focal input scored by
`flavi_scoring.ipynb`. Leaf labels are expected to start with a group code
and underscore. The notebook counts each internal node whose descendant
leaves all have the same group code, then divides that count by the total
number of internal nodes.

The directory also preserves comparison trees:

- E protein: `E_aa.nwk`, `E_af2.nwk`, `E_esm3di.nwk`, and
  `E_prostt5.nwk`.
- NS5: `NS5_aa.nwk`, `NS5_esm3di.nwk`, and `NS5_prostt5.nwk`.

All E and NS5 method comparisons are retained as monophyly experiments.
The matching section of `benchmark/real_tree_visualize_monophyly.ipynb`
explicitly labels the E inputs as an **E/E1/E2 mixture** and records the same
556/625 score. The filename `E_esm3di.nwk` alone does not establish an E-only
dataset; exact sequence composition requires checking the original inputs.

`folded_vsesm3di.ipynb` is another separate, exploratory all-against-all
Foldseek analysis of folded structures. It is not the featured E-protein
tree analysis. It points to the external Mifsud et al. structure collection
and uses absolute `/mnt/data2/...` paths; those source PDBs, search
databases, and intermediate alignments are not included here.

## Recorded envelope result

The saved output of `flavi_scoring.ipynb` reports 556 of 625 internal nodes
in `nwk/E_esm3di.nwk` as group-monophyletic (fraction 0.889600). This is a
notebook-recorded value, not a recomputed result.

## Retained method comparisons

The notebook also records scores for comparison trees:

| Protein | Tree input | Monophyletic / internal nodes | Fraction |
| --- | --- | ---: | ---: |
| E | AlphaFold | 490 / 625 | 0.784000 |
| E | ProstT5 | 521 / 625 | 0.833600 |
| E | Amino acid (MAFFT/IQ-TREE label) | 541 / 625 | 0.865600 |
| NS5 | Amino acid (labelled Mifsud) | 437 / 460 | 0.950000 |
| NS5 | ProstT5 | 434 / 460 | 0.943478 |
| NS5 | ESM3Di | 435 / 460 | 0.945652 |

These are retained exploratory monophyly comparisons, not independently
validated phylogenetic findings. The directory also contains `classic_foldtree.nwk` and
`esm3di_foldtree.nwk`; the scoring notebook does not score these files.
The tree-building notebook records use of
`checkpoints/esm3di_from_mlm/epoch_5.pt`, but historical checkpoints in this
series did **not** transfer MLM weights. Nothing here demonstrates an MLM
fine-tuning benefit.

## Reproduction and interpretation caveats

The focal monophyly score can be recalculated by running
`flavi_scoring.ipynb` from this directory with Biopython installed.
Rebuilding the trees requires
the omitted source alignments/structures and databases, Foldseek, the
external `fold_tree` module, FastME or QuickTree, and MAD rooting where
applicable; the notebook embeds machine-specific absolute paths.
The MAD call now uses the installed ESM3Di Python MAD wrapper instead of an
absolute native-binary path. Install `esm3di` before rerunning those cells;
an explicit `madrootpath` override must point to a Python MAD script. Rooting
uses the notebook's Python interpreter and validates the resulting Newick.
Saved notebook outputs and historical trees have not been regenerated.

The fraction is a label-purity summary of internal nodes, not a measure of
tree accuracy, support, or evolutionary model fit. It depends on group
prefixes, taxon sampling, tree shape, and how unresolved or missing taxa are
handled; neither the focal score nor comparison scores establish that a
tree recovers the true species history.

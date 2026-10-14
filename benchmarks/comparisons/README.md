# Alignment and fold-recognition benchmarks

This folder archives several research notebooks, not one reproducible
benchmark pipeline. The featured evidence represented here is the MetaVR 3Di
alignment comparison and SCOPe40 fold-recognition benchmark.
Monophyly scoring and large-tree visualization now live in the
[phylogeny experiment archive](../../experiments/analyses/phylogeny/README.md).

## Featured analyses

- **Alignment benchmark — `3di_alignment.ipynb`.** Compares
  AF3/Foldseek-derived, ProstT5-predicted, and ESM3Di-predicted 3Di for a
  10,000-entry MetaVR subset. It builds Foldseek databases, runs paired
  `structurealign` comparisons, and compares bitscores and per-sequence 3Di
  identity. Saved cell outputs confirm 10,000 entries and 10,000 comparison
  rows; plots are embedded in the notebook.
- **Fold-recognition benchmark — `scope_family.ipynb`.** Uses the viral
  SCOPe40 subset to compare family/fold assignment search results for
  ProstT5, Foldseek structure-derived 3Di, MMseqs2, and ESM3Di. It builds the
  subset, assigns SCOPe labels, runs all-vs-all searches, and plots AUROC1
  curves. The saved output records 16,272 viral entries identified during
  filtering, but does not preserve a concise numerical summary of the final
  curves.

## Supporting and other exploratory material

- **`3di_baseline.ipynb`** samples 3Di symbols independently according to
  frequencies counted in a BFVD 3Di FASTA, then compares the random sequences
  with MetaVR 3Di. The recorded run reports mean identity **14.7584%**. This
  is a frequency-only null baseline, not a trained-model result; no random
  seed is set, so the result is not exactly reproducible. Its positional
  pairing is not validated by record identifier, so this number is provisional.
- **`metavr_taxonomy.ipynb`** extracts an IMG/VR protein-cluster to UViG/ICTV
  taxonomy mapping. **`taxonomy_split.ipynb`** contains MetaVR and SCOPe40
  taxonomy split work. `clades.txt` is kept alongside the split notebook
  that reads it. These are supporting data-preparation materials, not
  independent benchmark results.

Notebook cell outputs are the only results retained here for these analyses.
Intermediate FASTAs, Foldseek databases, alignments, score tables, and
taxonomy/tree inputs were written to external working directories and are not
tracked in this folder.

## Inputs and reproduction

The notebooks expect external datasets and locally installed tools. Recorded
paths include `/mnt/data1/metavir_structs` and
`/mnt/data1/scop_benchmark`; the notebooks also use downloaded MetaVR/SCOPe
files, ICTV/NCBI taxonomy data, ProstT5 weights, Foldseek/MMseqs2, Python
packages for plotting and phylogeny, and (for ESM3Di prediction) a compatible
PyTorch/CUDA environment. Update paths and obtain the same versions and source
datasets before attempting reproduction; the repository does not pin all of
these inputs or package versions.

Install the repository in the notebook environment and run notebooks from
this directory. Relative repository checkpoint paths have been adjusted for
the new location. Absolute external dataset paths remain unchanged.
The prediction cell no longer changes into an obsolete hardcoded repository
directory; it uses the installed package from the notebook's working directory.
`scope_family.ipynb.bak` is an unchanged historical backup, not the runnable
notebook; its old paths and CLI syntax have not been migrated.

Some cells create or overwrite files in those external working directories.
For example, the MetaVR database-preparation cell removes existing contents of
its `$DATAPATH/foldseek` directory. Review paths and destructive cells before
running a notebook top-to-bottom. These notebooks are exploratory, include
manual preparation steps, and are not CLI recipes.

The ESM3Di prediction cells in `3di_alignment.ipynb` and `scope_family.ipynb`
refer to `checkpoints/esm3di_from_mlm/epoch_5.pt`. This historical checkpoint
did **not** actually load the MLM backbone weights despite its directory name.
Results made with it must not be presented as evidence of an MLM fine-tuning
benefit. Commented alternative checkpoint paths are not provenance for the
recorded outputs.

## Caveats

- Search settings, data subsets, tree construction, and taxonomic filtering
  differ across notebooks. Compare results only within the method and dataset
  specified by each notebook.
- A tree's monophyly fraction is a label-purity summary of internal nodes, not
  a measure of tree accuracy, support, or evolutionary model fit. It depends
  on group labels, taxon sampling, tree shape, and how unresolved or missing
  taxa are handled.
- Plots and notebook outputs do not by themselves establish statistical
  significance, independent test performance, or a causal advantage of a
  representation. Reproduction requires recovering external inputs and
  documenting the software environment.

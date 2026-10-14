# Flavivirus monophyly experiments and exploratory region analysis

## Biological question

The retained flavivirus topics in this archive include envelope-labelled
E/E1/E2 and NS5 tree comparisons in `simple_flavi/`: the question is whether
the inferred trees group labelled flavivirus taxa into monophyletic subtrees.
`simple_flavi/flavi_scoring.ipynb` reports a descriptive score for
`simple_flavi/nwk/E_esm3di.nwk`.

`clip_regions.ipynb` is a separate, exploratory all-against-all polyprotein
alignment workflow. It clips candidate regions, builds domain clusters, and
explores further trees. This is supporting archive material, **not** the
featured flavivirus tree analysis; it must not be described as a benchmark of
the E-protein tree or as evidence about that tree.

## Featured evidence

The saved scoring notebook reports 556 of 625 internal nodes in
`simple_flavi/nwk/E_esm3di.nwk` as group-monophyletic (fraction 0.889600).
This is a recorded output, not an independently validated rerun or a measure
of phylogenetic accuracy. The accompanying E-labelled trees (`E_aa.nwk`,
`E_af2.nwk`, and `E_prostt5.nwk`) and NS5 trees are retained comparisons.
The benchmark scoring notebook labels the matching envelope comparison as
an E/E1/E2 mixture. An independently confirmed E-only input set is not
established by these filenames or scores.

## Inputs

The separate `clip_regions.ipynb` workflow expects files that are not
included in this directory:

- `flavi_allvall.tsv`: Foldseek all-against-all results, with query/target,
  alignment coordinates, E-value, and bit score columns.
- `sequence_species_mapping.tsv`: mapping from sequence IDs to species.
- Original protein FASTAs for extracting the selected regions.
- Foldseek databases and reference structures for the domain comparisons.

It also refers to an ESM3Di checkpoint at
`../../../checkpoints_ESMplusplus_small/epoch_3.pt`. The only files currently
stored directly under this experiment directory are the notebook itself.

## Supporting exploratory workflow

The clipping notebook computes a table excluding self-hits and weak alignments
(bit score `> 50`, E-value `< 1e-5`), associates sequence IDs with species,
and groups overlapping alignment intervals into consensus regions using
overlap, species-support, and adjacent-position Jaccard thresholds. The
region call uses a 15% sequence-overlap threshold, support from at least 10
species, and Jaccard threshold 0.95. Despite a nearby comment saying “at
least 20 amino acids,” the executed filter retains regions of at least 100
amino acids. The implementation caveat below notes that the region call
uses the original alignments rather than the computed filtered table.

It writes `consensus_regions.tsv`, BED-like boundaries, and
`domain_annotations.tsv`; optional later cells extract domains, cluster them
using Foldseek similarity (the shown call uses identity threshold 0.2 and
coverage threshold 0.5), compare clusters with reference structures, build
Foldseek-derived trees with QuickTree, and pass gene trees to ASTRAL.

Two implementation details matter when interpreting the saved output:

1. The notebook computes a filtered alignment table, but the consensus-region
   call passes the original `foldseek_results`, not that filtered table.
2. Generated TSVs, FASTAs, cluster directories, and species-tree outputs are
   not checked in here. The numbers below are notebook-recorded outputs, not
   outputs regenerated from available input files.

## Recorded results from the clipping workflow (not featured evidence)

Saved notebook output reports 315,501 input alignments, 621 unique query
sequences, and a species mapping with 394 species for 621 sequences. It
reports 1,186 consensus regions across 610 sequences, with mean region
length 1,391.7 residues (SD 1,376.6) and mean coverage 287.9. A later
domain-clustering cell reports 66 clusters from a graph with 928 nodes and
26,851 edges. These are the recorded values in the notebook; the underlying
inputs and derived files are absent, so the results cannot be independently
recomputed from this directory.

## Reproduction and caveats

Reproduction needs the missing TSV/FASTA inputs, species mapping, reference
structures and checkpoint, Foldseek, QuickTree, ASTRAL (`astral4`), and the
notebook's Python dependencies. The tree-building cells also rely on an
external `fold_tree` module. Paths are mostly relative, so run from a
consistent working directory and update locations for the local checkout.

Alignment overlap and species counts are heuristic evidence of shared
regions, not proof of homology or functional domain boundaries. Results may
be sensitive to sequence sampling, duplicate sequences, alignment/search
settings, coordinate conventions, thresholds, and the noted filtered-table
inconsistency. Treat extracted domains and inferred trees as candidate
analyses requiring independent sequence and biological validation.

For its inputs, tree-scoring method, supporting E-tree scores, and caveats,
see [`simple_flavi/README.md`](simple_flavi/README.md).

# Ebola protein-family trees and supporting analyses

## Biological question

The featured Ebola topic is the set of saved Ebola protein-family trees for
glycoprotein (GP), polymerase (L), and nucleoprotein (NP). These trees
describe sequence/structure-similarity relationships among the archived
family sequences; they are exploratory and are not a formal phylogenetic or
taxonomic conclusion.

The notebook also contains supporting data-preparation and taxonomy steps,
plus a distinct exploratory Oberland virus translated-ORF search. That
Oberland search is **not** evidence from the featured Ebola trees.

## Featured evidence: Ebola trees

- `ebola_data/Ebola_{GP,L,NP}_taxonomic_info.csv` and matching
  `*_cdhit_taxonomic_info.csv` files contain accession-level taxonomy tables
  for the three protein families and their clustered counterparts.
- `ebola_data/ebola_protfams*/` contains stored Newick trees, including
  unclustered and clustered family trees. Some trees have corresponding
  `.pp.nwk` outputs.

The notebook describes building all-against-all Foldseek family comparisons
and converting them into trees. The Newick files are saved archive outputs;
they are not a validation of the tree-building workflow or a rerun from the
original FASTAs.

## Supporting archive material: Oberland hit search

- `oberland_virus_geno.fasta` is the nucleotide input used for six-frame
  translation.
- `ebola_data/oberland_results/best_hits_summary.csv` is the saved
  best-hit summary.

The source protein FASTAs, Foldseek databases, translated six-pack FASTA, and
many intermediate search files referenced by the notebook are not present in
this directory.

The supporting method optionally clusters family FASTAs at 90% identity
with CD-HIT and fetches taxonomy for accession identifiers. For the
Oberland genome it runs EMBOSS `sixpack`, retains long translated
sequences, searches those against the GP/L/NP Foldseek databases, filters
hits at E-value `<= 1e-5`, and saves up to ten top hits per query. Other
cells create FASTA/tree inputs for further inspection.

In the checked-in notebook, the CD-HIT and tree-generation switches are
initially `False`; existing trees and tables are saved historical artifacts,
not evidence that those stages run automatically from a clean checkout.

## Supporting recorded result: Oberland search

The saved `best_hits_summary.csv` contains 22 hit rows for three translated
query identifiers (`long_seq_2`, `long_seq_5`, and `long_seq_7`) across the
GP, L, and NP database variants. Its strongest listed hit is `long_seq_5`
against target `QGM12360.2` in an Ebola GP database (`fident` 1.0, alignment
length 618, E-value `1.385e-95`). This is a record in the saved search table;
the input ORF sequence, database provenance, and a fresh search have not been
independently validated here. The query name alone does not identify the
translated protein.

## Reproduction and interpretation caveats

Reproduction requires the original family FASTAs and Foldseek databases,
plus Foldseek, CD-HIT, EMBOSS `sixpack`, QuickTree (or FastME), and the
notebook's Python dependencies. Taxonomy retrieval also depends on external
NCBI services and accession availability. The notebook contains absolute
paths from its original machine (including `/home/dmoi/...`) and assumes
data below a relative `ebola_data/` directory; update these paths and
restore the missing inputs before rerunning. The checked-in results have not
been regenerated as part of this documentation.

The notebook uses sequence-similarity thresholds and a top-hit selection
rule, so hits depend on database coverage, sequence translation/frame
selection, clustering, and search settings. Similarity is not by itself
proof of orthology, function, or viral classification. Treat tree placement
and hit labels as leads for follow-up, not definitive biological conclusions.

# Pathogen Data Network sequence files

This directory is a collection of pathogen protein-sequence FASTAs and
companion entry-list text files. It is supporting archive data, not one of the
featured analyses (MetaVR alignment, SCOPe40 fold recognition, flavivirus
envelope/E tree, Ebola trees, or cleavage-site prediction). It is not a
self-contained analysis: there is no script or notebook here that specifies a
scientific comparison, generates results, or associates each FASTA record
with a model checkpoint.

## Contents

The FASTAs are grouped by virus family/genus labels in their filenames:

| FASTA | Records in the checked-in file |
|---|---:|
| `Hepacivirus.fasta` | 262 |
| `Pestivirus.fasta` | 252 |
| `orthoflavivirus.fasta` | 728 |
| `pegivirus.fasta` | 210 |
| `Additional_sequences.fasta` | 24,490 |

The headers carry sequence identifiers and, in examples, organism/isolate,
accession-like identifiers, sequence coordinates, and protein-product
annotations (for example, mature core protein or peptide 2k). Companion
`entries_fasta_*.txt` files contain flat-file-style records with `ID` and `AC`
fields. The observed number of `ID` records in those files is 26 for
Hepacivirus, 21 for Pestivirus, 56 for Orthoflavivirus, 21 for Pegivirus, and
2,449 for `entries_fasta_additional_sequences`. These counts are not a
one-to-one match to the FASTA record counts; no mapping procedure is included.

At least one FASTA record is only one amino acid long. Apply and document
sequence-quality checks before using these files in a model or benchmark.

## Intended use and reproduction limits

The filename `PDN_Data` suggests Pathogen Data Network material, but no
source URL, release/date, licensing terms, download or filtering script, or
record-level provenance is included here. The records appear to be amino-acid
sequences from annotated viral proteins, but the exact biological inclusion
criteria and the relationship between each entry list and FASTA are
undocumented. No evidence here ties these sequence sets to the cut-site labels
in `polyprotein_cut` or another featured analysis. There are no stored
analysis outputs or checkpoint references.

Reproducing a downstream result requires identifying the original data
release, resolving the entry-to-sequence mapping, validating sequence lengths
and identifiers, and recording any filtering or transformation. These files
are separate from the `polyprotein_cut` notebook inputs unless a downstream
analysis explicitly links them.

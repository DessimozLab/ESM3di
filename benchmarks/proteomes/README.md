# `benchmark_proteomes` database artifact

This directory contains a prebuilt sequence-search database named
`all_clades_db`. It is an archived supporting artifact, not one of the
verifiable featured results (MetaVR alignment, SCOPe40 fold recognition,
monophyly comparisons, or cut-site prediction). The
broader archive includes search work, but no notebook or script here documents
how this particular database was made or which experiment consumes it. Its
exact scientific role and construction therefore cannot be verified from the
stored files alone.

## Files and inspectable content

The database bundle consists of:

- `all_clades_db`, `all_clades_db.index`, and `all_clades_db.dbtype`
- `all_clades_db_h`, `all_clades_db_h.index`, and `all_clades_db_h.dbtype`
- `all_clades_db_ss`, `all_clades_db_ss.index`, and
  `all_clades_db_ss.dbtype`

The accompanying `.index` and `.dbtype` files, together with the `_h` and
`_ss` companions, identify these as MMseqs2/Foldseek-style database files.
The header database is readable and begins with viral protein accessions such
as SARS entries. These are generated binary/search artifacts, not FASTA
files. This partial header inspection does not establish clade coverage or
that the database is an input to either featured search benchmark. The file
names and headers also do not establish whether the `_ss` component is a
particular predicted-3Di representation.

No performance results, query set, model checkpoint, or per-clade summary are
stored here. The files are evidence of a database artifact only; they do not
support a claim about benchmark accuracy or sensitivity.

## Reproduction and provenance limits

The source FASTA, database-construction command, software version, filtering
rules, sequence counts, taxonomy snapshot, and checksums are not recorded in
this folder. No checkpoint provenance is stored with the database. To
reproduce or reuse it scientifically, first locate the original source
sequences and construction procedure, verify database compatibility with the
Foldseek/MMseqs2 version in use, and document any query database, search
parameters, and checkpoint separately. Do not infer the complete experiment
from this directory name alone.

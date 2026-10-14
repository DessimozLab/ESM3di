# HIV mutation and 3Di perplexity analysis

## Biological question

The featured topic is HIV mutation analysis: this notebook explores
per-residue 3Di-token perplexity across HIV protein variants and compares
those scores with experimental antiretroviral susceptibility measurements.
It also includes supporting visualization of substitution-associated
perplexity on selected HIV protease and reverse-transcriptase structures.

This is an exploratory model-scoring analysis, not a clinical resistance
predictor or a test of causality.

## Inputs

- `PI_DataSet.txt`, `NRTI_DataSet.txt`, and `NNRTI_DataSet.txt` contain
  isolate-level genotype and PhenoSense susceptibility data for protease and
  reverse transcriptase. `CAI_DataSet.txt` is the capsid mutation-screen
  table.
- The notebook embeds consensus sequences, writes them to
  `consensus_seqs.fasta`, and constructs mutant FASTAs from the tabular
  substitutions.
- `hivdb_structures/` contains reference PDB files used for structural
  visualization.
- The notebook loads `../../../checkpoints/esm3di_from_mlm/epoch_5.pt`.

Although the notebook's mapping mentions integrase, no integrase data file is
provided here and that entry is explicitly unset. Do not interpret this
directory as an integrase-inhibitor analysis.

## Method

The notebook summarizes mutation counts, constructs protein variants, and
uses `ESM3DiModel` to calculate per-position 3Di perplexity for native
consensus proteins and mutant sequences. It then summarizes substitution
scores by sequence position, compares those summaries with the susceptibility
columns in the source datasets, and maps selected scores onto reference
structures.

The saved `HIV_mutation_screen.tsv` records saturation substitutions by
protein position and alternate amino acid, with summary fields including
minimum, maximum, mean, variance, skewness, and kurtosis of perplexity.
`HIV_native_perplexity.tsv` stores native per-position scores.
`*_mutations_perplexity.tsv` files store per-position scores for variants
constructed from the phenotype datasets.

## Recorded artifacts and results

The TSV files listed above are present in this directory and contain
perplexity records. Their presence verifies that result tables were saved;
it does not validate a fresh rerun or establish a biological association.
The notebook's susceptibility comparisons and PDB overlays are supporting
exploratory analyses, not independently validated benchmark evidence. No
claims about a significant or predictive relationship are made here.

The checkpoint is named `esm3di_from_mlm`, but the historical checkpoints in
that series did **not** transfer the MLM weights. These files therefore do
not demonstrate a benefit from MLM fine-tuning.

## Reproduction and limitations

Run `HIV_LLManalysis.ipynb` in an environment with the repository's
`esm3di` code, its Python/model dependencies, PyTorch, and the referenced
checkpoint. The notebook relies on paths relative to its launch location and
on checkpoint files outside this directory; those paths may need adjustment
in another checkout. The inputs and PDBs here are not sufficient to reproduce
model scores without the matching checkpoint and software environment.

Perplexity measures model-assigned token surprise, not viral fitness,
resistance, or drug response directly. The phenotype tables are observational
and may contain missing measurements; correlations or structural coloring
should be treated as hypothesis-generating. Confirm numbering and chain
mapping before interpreting a residue on a PDB structure. The notebook's
assumption that sequence and PDB residue numbering align may not hold for
every structure.

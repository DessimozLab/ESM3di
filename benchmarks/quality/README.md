# 3Di quality benchmarks

Notebooks that measure how close predicted 3Di is to structure-derived 3Di.
The two main benchmarks draw their figures with the shared [plotting code](../README.md#shared-plotting)
and write the per-item results that the [paper figures](../paper/README.md) use.

## MetaVR 3Di alignment — [3di_alignment.ipynb](3di_alignment.ipynb)

For 10,000 MetaVR proteins with AlphaFold3 models, four 3Di representations
are compared:

- **AF3**: Foldseek 3Di from the AF3 structures (reference).
- **ProstT5**: Foldseek `createdb --prostt5-model` on the amino-acid sequences.
- **ESM3Di**: `esm3di foldseek-db` on the same sequences.
- **Random baseline**: 3Di sampled independently from BFVD 3Di letter
  frequencies ([bfvd_3di_freq_matrix.tsv](bfvd_3di_freq_matrix.tsv), written by
  the notebook), one sequence per protein with the protein's length.

ProstT5 and ESM3Di are aligned 1-vs-1 to the AF3 entry of the same identifier
with Foldseek `structurealign`, giving a bitscore. 3Di identity is the
percentage of positions with the same 3Di letter as AF3.

Output: [results/3di_benchmark.csv](results/3di_benchmark.csv) (one row per
protein: `query`, ProstT5/ESM3Di bitscore, ProstT5/ESM3Di/baseline identity)
and the figures in [figures/](figures).

| 10,000 proteins | ProstT5 | ESM3Di | Random baseline |
|---|---|---|---|
| Median 3Di identity (%) | 57.96 | 60.29 | 14.01 |
| Median bitscore | 647 | 663 | – |

This notebook replaces the earlier `benchmarks/comparisons/3di_alignment.ipynb`
and `benchmarks/comparisons/3di_baseline.ipynb` (available in git history). The baseline is
now computed for the same proteins and matched by identifier, but the sampling
still has no random seed, so baseline values are not exactly reproducible.

## SCOPe40 viral fold recognition — [scope_family.ipynb](scope_family.ipynb)

Reproduces the SCOPe benchmark of the
[Foldseek paper](https://www.nature.com/articles/s41587-023-01773-0) on the
viral subset of SCOPe40 (2.08). It downloads SCOPe and the NCBI taxonomy,
selects viral domains, removes multi-model entries, and runs all-vs-all
searches for ProstT5 3Di, ESM3Di 3Di, Foldseek on the real structures, and
MMseqs2-style sequence alignment. For each query, sensitivity is the fraction
of true positives at the family, superfamily and fold level ranked before the
first false positive (a hit from a different fold).

Output: [results/scope_sensitivity.csv](results/scope_sensitivity.csv) (one row
per query, level and method) and the figures in [figures/](figures).

This notebook replaces the earlier `benchmarks/comparisons/scope_family.ipynb`.

## MLM versus 3Di-only fine-tuning — [3di_alignment_mlm_vs_3di_only.ipynb](3di_alignment_mlm_vs_3di_only.ipynb)

The same MetaVR alignment benchmark, used to compare two ESM3Di checkpoints
rather than ESM3Di with ProstT5. Executed on 2026-10-15, comparing the
historical effectively 3Di-only `esm3di_from_mlm/epoch_5.pt` with the actual
MLM-initialized `esm3di_from_mlm_fixed/epoch_5.pt`. Both were rerun with
identical inference settings on all 10,000 MetaVR entries, matched by
identifier against the same AF3 reference. Mean per-protein 3Di identity was
**60.7938% versus 60.3944%**, and mean alignment bitscore was **1112.7748
versus 1104.3969** (3Di-only versus MLM + 3Di). Thus MLM initialization did not
improve these aggregate benchmark metrics. This is not a controlled causal
ablation: training code/settings also differ, including training truncation.
The notebook preserves outputs, paired plots, paths, and software/input
hashes. It uses absolute paths on the machine where it was run; raw artifacts
are under `/mnt/data1/metavir_structs/esm3di_mlm_comparison_20261015/`.

The historical `esm3di_from_mlm` checkpoints did **not** actually load the MLM
backbone weights despite their directory name. Results made with them must not
be presented as evidence of an MLM fine-tuning benefit.

## Running

Run the notebooks from this directory with the repository and Foldseek
installed. `DATAPATH` points to an absolute scratch directory and the BFVD 3Di
FASTA path is machine-specific; set both before running. The MetaVR cells
download the first 2 GB of the MetaVR structure archive and sample 10,001 of
the extracted structures with a fixed seed.

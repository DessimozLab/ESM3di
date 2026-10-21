# Analysis and experiment index

These are research records rather than independently packaged pipelines.
Each linked README describes the question, available inputs, methods, outputs,
and limitations supported by the archived files. Consult it before interpreting
saved notebook figures or attempting a rerun.

| Experiment or archive | Documentation |
|---|---|
| Structural classification and alignment benchmarks | [quality](../../benchmarks/quality/README.md) |
| Speed benchmarks | [speed](../../benchmarks/speed/README.md) |
| Proteome-scale search material | [proteomes](../../benchmarks/proteomes/README.md) |
| Monophyly scoring, large-tree visualization and taxonomy preparation | [phylogeny](phylogeny/README.md) |
| HIV mutation analyses | [HIV_mutations](HIV_mutations/README.md) |
| Ebola protein analyses | [ebola_data](ebola_data/README.md) |
| Flavivirus analyses | [flavi_redo](flavi_redo/README.md) |
| Polyprotein cleavage prediction | [polyprotein_cut](polyprotein_cut/README.md) |
| Fast structure-aware alignment and tree construction | [fasttree_NJ](fasttree_NJ/README.md) |
| PDN data archive | [PDN_Data](PDN_Data/README.md) |
| Historical training configurations | [legacy_configs](legacy_configs/README.md) |
| Batch training and evaluation launchers | [scripts](scripts/README.md) |
| Archived training curves | [checkpoints_notebook](checkpoints_notebook/README.md) |
| Scratch outputs and exported remote code | [scratch](scratch/README.md) |

## Interpreting historical results

The old 3Di training runs named `esm3di_from_mlm` did **not** load the intended
MLM backbone weights. Their results can characterize the old trained models,
but cannot establish that preceding 3Di training with MLM improved performance.
The fixed MLM-to-3Di rerun uses separate checkpoint directories and must be
evaluated independently.

Saved notebook outputs are historical observations, not newly reproduced
results. Many analyses depend on local databases, external tools, absolute
paths, or checkpoints that are not part of this archive. Moving research files
does not make those dependencies portable.

For current supported usage, see the [repository README](../../README.md).

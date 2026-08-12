# Viral-FoldTree (ESM3di Variant)

A Snakemake pipeline that infers structural phylogenetic trees directly from raw amino acid sequences.
!! In development, not tested !!

## 🔄 Workflow Logic
1. **Sequence Translation:** Converts raw AA sequences into structural 3Di strings using **ESM3di** (bypassing the need for explicit 3D PDB generation).
2. **FoldTree Core Execution:** Feeds the 3Di strings directly into the standard FoldTree pipeline framework for structural alignment (**Foldseek**), matrix calculation, and tree building.
3. **Phylogeny Rooting:** Finalizes the pipeline by applying **MAD** (Minimal Ancestor Deviation) rooting to the structural trees.

## 🚀 Quick Start
1. Configure model weights, execution resources, and local binary paths in `config/config.yaml`.

2. Run a dry-run to verify the DAG:
```bash
   snakemake -np --use-conda
```

3. Execute the workflow:
```bash
snakemake --use-conda --cores 2
```



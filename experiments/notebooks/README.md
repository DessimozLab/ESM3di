# Notebook archive: supporting and exploratory material

These notebooks document historical demonstrations and analyses, not the supported training recipe. For current usage and checkpoint compatibility, see the repository [README](../../README.md), [configs](../../configs/), and the supported `esm3di` CLI. The supported two-stage workflow is **ESM++ MLM fine-tuning, then supervised 3Di training initialized from that MLM backbone before LoRA and the CNN head are applied**. The notebooks below include older and experimental paths; their code, dependencies, external data paths, and checkpoints may need adaptation.

## Featured analyses versus this folder

The retained scientific themes in the broader experiment archive are: **all monophyly experiments**, including flavivirus NS5, envelope-labelled E/E1/E2 comparisons, and Ebola trees; alignment and fold-recognition benchmarks; viral polyprotein cut-site prediction; and HIV mutation analysis. This notebooks folder is supporting or other exploratory material rather than primary evidence for those themes. In particular, `comparative_tree_analysis.ipynb` sets up broader structure-derived/predicted 3Di and sequence comparisons, all-vs-all FoldSeek searches, and reference-structure clustering. Do not conflate that workflow with the scored monophyly trees. The other notebooks address model demonstration/evaluation, an older training prototype, and track visualization.

## `ESM3Di_Colab.ipynb` — FASTA inference and FoldSeek hand-off

**Supporting usage demonstration; not one of the featured analyses.** An interactive Colab-style walkthrough for uploading amino-acid FASTA sequences, loading a selected ESM3Di checkpoint, predicting per-residue 3Di tokens, viewing/downloading FASTA results, and optionally building a FoldSeek database for downstream search and FoldTree-style analysis. The notebook emphasizes viral proteins because the described training data are BFVD. Its discussion of downstream evolutionary analysis is a proposed use case, not a result reported by this notebook. Check model IDs, checkpoint compatibility, and current CLI guidance before using it; old `esm3di_from_mlm` checkpoints did not in fact inherit the MLM weights (see the [loss report](../reports/training_validation_loss_report.md)).

## `comparative_tree_analysis.ipynb` — structure, predicted 3Di, and sequence comparison

**Other exploratory tree/comparison work; not the featured flavivirus tree.** Sets up a comparison involving structure-derived 3Di, ESM3Di-predicted 3Di, and amino-acid sequences, with FoldSeek and external alignment/tree tools named in the introduction. The cells also create FoldSeek databases, run all-vs-all searches, and explore reference-structure clustering and cluster FASTAs. This is not a single phylogeny built from the flavivirus envelope protein, and the checked-in cells do not document a complete, validated tree comparison or its results. The example uses local structure/reference paths and a checkpoint path that must be supplied for a particular dataset. If substituted with a historical `esm3di_from_mlm` checkpoint, do not interpret results as evidence for an MLM-to-3Di transfer benefit.

## `evaluate_model.ipynb` — checkpoint and taxon evaluation

**Supporting model evaluation; not a featured biological benchmark.** Builds a small evaluation set from BFVD, loads amino-acid and 3Di FASTAs, evaluates selected checkpoints, and organizes comparisons including accuracy distributions, confusion matrix, class-level performance, example predictions, pLDDT-stratified results, ProSTT5, and cross-taxon models. Several later cells analyze pLDDT distributions and per-class performance. The notebook contains narrative interpretations, but these should be treated as hypotheses to verify against the exact inputs, checkpoint provenance, and outputs used for a given run—not as universally established results. In particular, comparisons involving legacy `esm3di_from_mlm` checkpoints are not MLM-initialized comparisons: those checkpoints did not load the MLM backbone weights.

## `retrain_experiment.ipynb` — earlier ESM-2/LoRA classifier prototype

**Other exploratory training prototype.** Implements a standalone sequence dataset and collator, loads an ESM-2 model with LoRA for per-residue 3Di classification, trains with a custom loop, plots losses, and demonstrates checkpoint loading and inference. It is an exploratory ESM-2-era implementation, not the supported ESM++ MLM→LoRA/CNN workflow. The notebook's own next steps call for validation and further experimentation; its training-loop examples should not be read as evidence of generalization or as a recipe for reproducing the supported pipeline.

## `visualize_tracks.ipynb` — experimental per-residue tracks

**Other exploratory structural-track visualization.** Calls the experimental `esm3di.experimental.tracks` utilities on a structure, visualizes discrete tracks (amino acid, pLDDT, contact, range, burial, bend, and torsion) as a residue-by-track heatmap, and colors a structure with `py3Dmol`. The example references a local structure path outside this directory, so that input must be provided. Track visualization is experimental and separate from the supported 3Di token-classification stages.

## Example structure file

`AF-P0CG47-F1-model_v4.pdb` is a structure data file stored alongside the notebooks; it is not itself a notebook or evidence of a model evaluation.

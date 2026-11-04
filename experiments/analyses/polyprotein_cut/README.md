# Viral polyprotein cleavage-site comparison

`cut_flavi.ipynb` and `run_comparison.py` evaluate whether the 3Di-trained ESM3Di representation transfers to ICTV-annotated viral cleavage sites better than frozen stock ESM++ embeddings. This is a representation comparison, not a claim to be a best-in-class cleavage predictor.

## Experiment

- **Embeddings:** ESM3Di from `checkpoints/esm3di_from_mlm/epoch_5.pt` versus stock `Synthyra/ESMplusplus_small` at revision `46c5f7d`. The historical checkpoint name is misleading: it was not initialized from MLM weights.
- **Labels and sequences:** exact protein sequences from the four ICTV flat files, with cut positions parsed by the notebook's original rule (`FT SITE` annotated `/note="Cleavage"`, using the first residue of each range). Only proteins with parsed cleavage labels are included.
- **Outer evaluation:** leave one group out across Hepacivirus, Orthoflavivirus, Pegivirus, and Pestivirus. Hyperparameters are fixed before the outer evaluation; no held-out group is used for model selection.
- **Heads:** the same FFN and sequence Transformer are trained on each representation, with the same fixed random seed (17) for each paired run and eight epochs. The Transformer head is trained on 512-residue chunks with 64-residue overlap; overlap supplies context while its labels are masked from duplicate loss.
- **Metrics:** ROC AUC, strict residue-level AP, residue-level AP after expanding labels by ±2 and ±5 residues, and macro per-protein top-10 site recovery with non-maximum suppression and one-to-one matching within ±5 residues.
- **Long proteins:** both embedding sources use identical 1024-token overlapping windows; overlap embeddings are averaged so no residues are dropped.

The summary has one row per held-out group, embedding source, and head. `seed_metrics.csv` has the run-level values; only one seed is currently used, so seed variability is not estimated. `oof_ffn_predictions.csv` stores per-residue out-of-fold FFN scores for ROC analysis. `figure2_ap_by_group.png` plots strict AP by group for both embedding sources. `run_metadata.json` records checkpoint hash, seed list, device, and metric definitions.

## Data audit

The former notebook evaluated 231 alignment rows, including repeated protein identifiers and proteins with no parsed cleavage annotations; missing annotations had been converted to all-negative masks. This comparison uses the 86 unique ICTV records with parsed cleavage labels instead. Under the existing label parser these contain 800 unique positive sites and 287,411 residues. This changes the evaluation set from the historical notebook results, which should not be compared directly.

## Running

On this machine, use its ESM3Di Blackwell environment from the repository root:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 /home/dmoi/.venvs/esm3di-blackwell/bin/python \
  experiments/analyses/polyprotein_cut/run_comparison.py
```

The workspace sandbox hides `/dev/nvidia*`; run GPU jobs in a host execution context where `nvidia-smi` and PyTorch can see the GPUs. The script caches embeddings under `/tmp/polyprotein_cut_esm3di_vs_esmpp` and writes results under `results/esm3di_vs_esmpp/`. The notebook uses the `ESM3Di Blackwell` kernel and displays paired metrics, the AP figure, and FFN ROC curves overall and by held-out group.

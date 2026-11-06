# Cut-site analysis: rework brief

Input for an agent reworking `experiments/analyses/polyprotein_cut/cut_flavi.ipynb`
(ESM3Di repo), to turn the cleavage-site results into a Figure 2 panel for the
ESM3Di brief communication.

## Why this is being asked for

The paper's phylogeny results do not separate ESM3Di from ProstT5, or from trees
built on real AlphaFold2 structures. So the paper has no result showing an
advantage on anything except 3Di token accuracy, where the margin is ~2 points.

The cleavage task tests a different claim: that fine-tuning on viral structures
improves the **representation** in a way that transfers to viral tasks beyond
3Di prediction. Nobody has shown this for a viral-tuned pLM, and it is also the
answer to Phold (NAR 2026), where fine-tuning ProstT5 on ~1M phage structures
gave "modest improvements" and the baseline model was shipped instead.

The claim needs exactly one comparison: ESM3Di embeddings against non-fine-tuned
ESM++ embeddings, same pipeline, same splits, same heads.

## Scope of the claim

The panel is **not** a cleavage-prediction tool claim. Dedicated cut-site
predictors exist (3CLP for coronavirus 3CL protease; `clear-bio/ingmar`, a
pip-installable pipeline on frozen ESMC-300M), and competing with them is out of
scope and not the point.

The claim is narrower and self-contained: **ESM3Di embeddings outperform
non-fine-tuned ESM++ embeddings on a viral task that is not 3Di prediction.**
That requires exactly one baseline (base ESM++), and nothing else. Motif
baselines and comparisons to protease-specific tools are not needed.

Keep the framing in that lane. Claiming a best-in-class cleavage predictor pulls
in prior art the panel does not need to survive.

## Current state

Dataset: 4 groups in *Amarillovirales* (Hepacivirus, Orthoflavivirus, Pegivirus,
Pestivirus). 231 sequences in the embedding frame, 86 with parsed cleavage
annotations. 775,319 residue positions, 940 positives (0.12%).

| Evaluation | Model | ROC AUC | AP |
|---|---|---|---|
| Random residue-level split | FFN | 0.993 | 0.217 |
| Random residue-level split | Transformer | 0.988 | 0.201 |
| Held-out Orthoflavivirus (final) | FFN | 0.917 | 0.0669 |
| Held-out Orthoflavivirus (final) | Transformer | 0.914 | 0.0329 |

Held-out split: train 70 seqs (226,647 tokens, 319 pos), test 161 Orthoflavivirus
seqs (548,672 tokens, 621 pos, rate 0.113%). AP 0.067 is ~59x chance.

## Problems to fix

**1. Leaked within-group number — blocking.**
The FFN split (`cell 10`, `rng.shuffle(idx)` over flattened residue indices) puts
residues from the same polyprotein in both train and validation. These are
homologous flaviviral polyproteins with near-identical site positions, so AP
0.217 measures leakage. The draft's claim of "higher performance on closely
related viruses" rests on it and cannot stand.
-> Split by sequence, not by residue.

**2. Hyperparameters selected on the test group.**
`cell 18` sweeps regularisation configs ranked by held-out Orthoflavivirus AP;
`cell 19` then reports on the same group.
-> Fix hyperparameters a priori, or select on one group and report on another.

**3. Only one held-out clade.**
Orthoflavivirus is 161/231 sequences, so training used 30% of the data with 319
positives. The draft says "clades", plural.
-> Leave-one-group-out over all four groups; report per-group values and spread.

**4. No base-model baseline — this is the centrepiece, not a check.**
The notebook compares FFN vs Transformer only. Nothing shows the fine-tuning
mattered, which is the entire claim the panel exists to make.
-> Rerun the identical pipeline on frozen embeddings from non-fine-tuned
`Synthyra/ESMplusplus_small` (revision `46c5f7d`, same base model as the current
run), on identical splits and with identical heads and hyperparameters.

If base ESM++ matches ESM3Di, the panel should not go in the paper. That is a
valid outcome, not a failure.

**5. Scoring is too strict and probably understates the model.**
AP is computed at exact single-residue resolution: a prediction at 374 when the
annotated site is 373 counts as both FP and FN. Sites were annotated from
alignments, so that precision is not meaningful.
-> Add a tolerance window (+/-2 and +/-5 residues) alongside the strict number.
-> Add a use-case metric: fraction of true sites recovered in the top-k
predictions per polyprotein, k = expected site count (~10 here). Nothing
currently measures whether the tool separates a polyprotein into its cleaved
products, which is what the paper claims it does.

ROC AUC 0.917 with AP 0.067 is the signature of correct ranking scored strictly,
so this may change the picture most.

## Required outputs

One row per (held-out group x embedding source x architecture):

- group, embedding source (esm3di / esm++ base), model (ffn / transformer)
- ROC AUC, AP strict, AP +/-2, AP +/-5
- top-k site recovery
- n test sequences, n test positives, positive rate

Plus a figure-ready panel: PR curves or per-group AP for both embedding sources
on the same axes.

## Constraints and gotchas

- Checkpoint used: `checkpoints/esm3di_from_mlm/epoch_5.pt`. Despite the name
  that run did **not** load MLM backbone weights — it is effectively the 3Di-only
  checkpoint, and the better-scoring one (60.79% vs 60.39% identity). See
  `benchmarks/quality/README.md`. Keep it; do not describe it as MLM-initialised.
- Hard-coded path in the notebook: `/home/dmoi/projects/ESM3di/polyprotein_cut/
  dataset/Amarillovirales`. Point it at the in-repo `dataset/Amarillovirales`.
- Saved checkpoints in `models/` were retrained on all 231 sequences, so they are
  not the held-out models that produced the reported metrics.
- `cell 23` applies OSCAR to input weights; `cell 24` concludes it did not help.
  No need to revisit.

## Cost

Classifiers are MLPs over cached embeddings; the existing 4-config sweep ran in
92 seconds. Recomputing embeddings for 231 sequences with base ESM++ is
minutes on a GPU. Total 4 groups x 2 architectures x 2 embedding sources = 16
short runs. Roughly half a day, no model retraining.

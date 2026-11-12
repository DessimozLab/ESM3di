# Figure legends

Copied from the manuscript (`viral_struct_phylo`); keep the two in sync.

**Figure 1 | ESM3Di architecture and benchmarks.** **a** ESM3Di model architecture. **b** The 3Di sequences inferred by
ESM3Di are more consistent with AlphaFold3-derived 3Di sequences than ProstT5-inferred ones (MetaVR subset). **c** ESM3Di
enables viral protein domain homology search at a performance closer to experimental structure-derived search than
previous approaches, based on SCOPe family-level classification. **d** Runtime analysis on one A100 GPU shows ESM3Di runs
more than two orders of magnitude faster than ColabFold at 300 sequences, and 3.7-fold faster than ProstT5 at 10,000
sequences.

**Figure 2 | ESM3Di improves viral structural phylogenetics and polyprotein cut-site prediction.** **a** *Filoviridae*
glycoprotein structural trees from ESM3Di and ProstT5, from one sequence per species (left) and all 550 unique sequences
(right). ESM3Di reconstructs more accurate species trees when compared to ProstT5. **b** *Flaviviridae* E2 glycoprotein
structural trees from ESM3Di and AlphaFold2. ESM3Di reproduces structural trees comparable to the approach presented in
ref.18, while bypassing the expensive folding step. In **a** and **b**, tables give monophyly (higher is better) and in
**b**, ultrametricity (lower is better; see Methods). **c** Structural tree of a metagenomic viral protein cluster with
124,949 leaves, colored by the class of the bacterial host. **d** Cut-site prediction in *Amarillovirales* polyproteins
from frozen per-residue embeddings of ESM3Di (solid) and of the unmodified ESM++ backbone (dotted), showing ESM3Di
improves accuracy of cut-site predictions in all 4 genera.

**Supplementary Fig. 1 | LORA explanatory diagram.** **a**; simplified transformer encoder layer can be thought of as a
first part composed of multi head attention followed by a feed forward network block. This allows all of the positions
of the input to serve as context for the formulation of the output. **b**; LORA matrices are shown alongside the fixed
weight matrices in a simplified schematic of a transformer encoder layer. Activation functions are not shown and the
diagram shows the simplest version of self attention with just one attention head. The LORA matrices ( shown in red )
are created by multiplying two lower rank matrices of rank R. They are added to the weight matrices ( shown in blue )
before multiplication with the data matrices ( shown in white ). The feed forward block is shown as a single layer in
this diagram for the sake of simplicity. Blue arrows show the flow of input data through the transformer encoder layer
through successive matrix multiplications. The only model weights being updated during LORA training are the low rank
red matrices which contain fewer parameters than the blue model weight matrices. This allows for fine tuning large
models such as ESMc on consumer hardware.

**Supplementary Fig. 2 | Additional 3Di and SCOPe benchmarks.** **a**; Foldseek alignment bitscores of ESM3Di vs.
ProstT5 predictions against AlphaFold3-derived 3Di (MetaVR subset, n=10,000). **b**; Distributions of 3Di identity to
AlphaFold3-derived 3Di for ESM3di, ProstT5 and the random baseline; dashed lines show medians. **c**; Sensitivity up to
the first false positive at SCOPe superfamily (left) and fold (right) level (viral subset; Foldseek: experimental
structures).

**Supplementary Fig. 3 | Runtime as a function of sequence length.** Runtime of ESM3Di and ProstT5 on one A100 GPU
across ten sequence-length bins from approximately 100 to 1,000 residues. Each bin contains 32 viral Swiss-Prot
proteins, each repeated 16 times (512 sequences per run). Filled markers: end to end, including model loading. Open
markers: inference only, with the model already loaded (measured for ESM3Di only). Points show the mean of five
repetitions after a warm-up run.

**Supplementary Fig. 4 | *Filoviridae* glycoprotein (GP) trees without X-rich sequences.** Trees from ESM3Di (left) and
ProstT5 (right) 3Di of the 417 of 550 unique GP sequences with at most 5% unknown residues (X), colored by species as
shades of one hue per genus. The two trees share one scale. Monophyly as in Fig. 2.

**Supplementary Fig. 5 | *Flaviviridae* glycoprotein trees.** E, E1 and E2 glycoprotein trees from ESM3Di 3Di (left)
and 3Di of predicted structures (right; AlphaFold2, and ESMFold for 51 of the 247 E structures), colored by the viral
groups of Mifsud et al. One scale per row. Sequences without a group were excluded before tree building, as in Fig. 2b;
scores as in Fig. 2. Robustness of these scores is given in Supplementary Fig. 6.

**Supplementary Fig. 6 | Jackknife of the *Flaviviridae* glycoprotein trees.** How far the comparison between ESM3Di
and AlphaFold2 3Di trees depends on the sequences chosen. In each of 100 replicates a random 20% of the sequences was
removed and trees were built for both methods from the same sequences, so every replicate gives a paired difference and
the effect of the sequence choice largely cancels. Each box is the distribution of those 100 differences, ESM3Di minus
AlphaFold2: the box spans the interquartile range, the line in it is the median and the whiskers reach the 2.5th and
97.5th percentiles, covering 95% of the replicates. The taller black line is the same difference between the trees of
all sequences, the trees of Fig. 2b and Supplementary Fig. 5, where the scores themselves are given. **a**, Monophyly,
as in Fig. 2: a difference right of the dashed line favours ESM3Di. **b**, Ultrametricity, as in Fig. 2 (Methods): a
difference left of the dashed line favours ESM3Di, since a lower value is more ultrametric. Sequences without a group
were excluded before tree building, as in Fig. 2b. None of the differences between the trees of all sequences is
statistically significant (delete-d jackknife z-test, all p > 0.5, Holm-corrected for six tests). ESM3Di and AlphaFold2
trees are therefore statistically indistinguishable in both monophyly and ultrametricity on these data.

**Supplementary Fig. 7 | Metagenomic tree of MetaVR sequence clusters with structural homology.** The tree visualizes
the structural relationships among 127,135 viral proteins from 63 MetaVR sequence-based clusters, whose representatives
were grouped into a single structural cluster by Foldseek. Branches are colored according to their originating
sequence-based clusters, with the 10 most largest clusters shown in distinct colors, and the remaining 53 clusters shown
in grey.

**Supplementary Table 1 | Pairwise amino-acid sequence identities of the *Filoviridae* and *Flaviviridae* datasets.**
Each row summarises all pairwise identities within one set of sequences. Sequences of each set were aligned with MAFFT
(v7.525, `--auto`, single thread), and identity was computed for every pair as the percentage of alignment columns at
which both sequences carry a residue and the residues match. *Flaviviridae* rows cover the E, E1 and E2 glycoprotein
datasets used for the trees in Fig. 2b and Supplementary Fig. 3. *Filoviridae* rows cover the GP dataset of
Supplementary Fig. 2: "representatives" are the 14 species representatives used for the species trees of Fig. 2a, split
into pairs from different genera and pairs from the same genus; "all" are the 550 deduplicated GP sequences, split by
whether a pair comes from the same species, from different species of one genus, or from different genera. The last
column gives the share of pairs below 35% identity, the upper bound of the twilight zone, below which sequence-based
homology inference and alignment become unreliable. Note that the identities among all 550 sequences reflect GenBank
sampling effort, which is heavily concentrated on a few outbreak-associated species, rather than the diversity of the
family.

**Supplementary Table 2 | Polyprotein cleavage-site prediction from ESM3Di and ESM++ embeddings.** Per-residue cut-site
classification in *Amarillovirales* polyproteins, leave-one-genus-out over the four annotated genera. The same
classifier, hyperparameters, splits and seeds were used for both embedding sources; only the input features differ.
n seq and n sites give the size of each held-out fold. Values are means ± s.d. over five training seeds. AP, average
precision at single-residue resolution; AP ±2 and AP ±5, average precision with every residue within that distance of
an annotated site counted as positive; top-10, fraction of annotated sites matched one-to-one within ±5 residues by the
ten highest-scoring peaks per protein. Cleavage sites make up 0.28% of residues, so chance average precision is ≈0.003.

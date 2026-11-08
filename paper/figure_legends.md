# Figure legends (draft)

Names used throughout: ESM3Di; ProstT5; *Filoviridae* glycoprotein (GP); *Flaviviridae* E, E1 and E2 glycoproteins.

**Figure 1 | ESM3Di architecture and benchmarks.** **a**, Model architecture. **b**, 3Di sequence identity of ESM3Di vs
ProstT5 predictions to AlphaFold3-derived 3Di (MetaVR subset). Dashed lines: random baseline median. **c**, Sensitivity up
to the first false positive at SCOPe family level (viral subset), with Foldseek and MMseqs2 as references. **d**, Runtime vs
number of sequences on one A100 GPU. Filled markers: end to end, through a CLI call including model loading. Open markers:
inference only, with the model already loaded (measured for ESM3Di only). ColabFold ran up to 300 sequences; the dashed line
extrapolates a power-law fit to 10,000 sequences.

**Figure 2 | Structural phylogenies from ESM3Di 3Di.** **a**, *Filoviridae* glycoprotein trees from ESM3Di (top) and
ProstT5 (bottom) 3Di: one representative sequence per species (left; genus abbreviated to its initial) and all 550 unique
sequences (right), both coloured by genus with species as shades of its hue. On the representative trees, numbers at the
nodes of the established genus relationships give the percentage of 50 random choices of representatives whose tree
recovers that relationship; a relationship the tree does not recover has no node and so no number, as for
*Orthomarburgvirus* with *Dianlovirus* in the ProstT5 tree (Methods). **b**, *Flaviviridae*
E2 glycoprotein trees from ESM3Di 3Di (top) and from 3Di of the AlphaFold2 structures (bottom), coloured by viral group
(Mifsud et al.); Supplementary Fig. 4 gives the robustness of these scores to the choice of sequences. **c**, Structural
tree of a metagenomic viral protein cluster, coloured by the class of the bacterial host. **d**, [Reserved: polyprotein
cleavage results.] In **a** and **b**, the two trees compared share one scale and are rotated or mirrored for display only.
Scores at the top of each tree give monophyly (monophyletic internal nodes / all internal nodes) and, in **b**,
ultrametricity (Methods).

**Supplementary Fig. 1 | Additional 3Di and SCOPe benchmarks.** **a**, Foldseek alignment bitscores of ESM3Di vs ProstT5
predictions against AlphaFold3-derived 3Di (MetaVR subset, n = 10,000). **b**, Distributions of 3Di identity to
AlphaFold3-derived 3Di for ESM3Di, ProstT5 and the random baseline; dashed lines show medians. **c**, Sensitivity up to the
first false positive at SCOPe superfamily (left) and fold (right) level (viral subset; Foldseek: experimental structures).
The fold panel shows the first 10% of queries; the AUC is over all queries.

**Supplementary Fig. 2 | *Filoviridae* glycoprotein (GP) trees without X-rich sequences.** Trees from ESM3Di (left) and ProstT5 (right) 3Di of the 417 of 550 unique GP sequences with at most 5% unknown residues (X), coloured by species as shades of one hue per genus. The two trees share one scale. Monophyly as in Fig. 2.

**Supplementary Fig. 3 | *Flaviviridae* glycoprotein trees.** E, E1 and E2 glycoprotein trees from ESM3Di 3Di (left) and 3Di of
predicted structures (right; AlphaFold2, and ESMFold for 51 of the 247 E structures), coloured by the viral groups of Mifsud et al. One scale per row.
Sequences without a group were excluded before tree building, as in Fig. 2b; scores as in Fig. 2. Robustness of these scores is given in Supplementary Fig. 4.

**Supplementary Fig. 4 | Jackknife of the *Flaviviridae* glycoprotein trees.** How far the comparison between ESM3Di and
AlphaFold2 3Di trees depends on the sequences chosen. In each of 100 replicates a random 20% of the sequences was removed
and trees were built for both methods from the same sequences, so every replicate gives a paired difference and the
variation between replicates cancels. Each box is the distribution of those 100 differences, ESM3Di minus AlphaFold2: the
box spans the interquartile range, the line in it is the median and the whiskers reach the 2.5th and 97.5th percentiles,
covering 95% of the replicates. The taller black line is the same difference between the trees of all sequences, the trees
of Fig. 2b and Supplementary Fig. 3, where the scores themselves are given. **a**, Monophyly, as in Fig. 2: a difference
right of the dashed line favours ESM3Di. **b**, Ultrametricity, as in Fig. 2 (Methods): a difference left of the dashed
line favours ESM3Di, since a lower value is more ultrametric. Sequences without a group were excluded before tree building,
as in Fig. 2b and Supplementary Fig. 3. A box and its whiskers covering the dashed line mean the two methods are not
separated by that measure on that glycoprotein; these are descriptive spreads and not significance tests, because
replicates share most of their sequences and related viruses are not independent.

**Supplementary Fig. 5 | Protein clusters in a metagenomic structural tree.** The tree is coloured by IMG/VR protein cluster: the ten clusters shown and all other clusters (grey).

**Supplementary Fig. 6 | Pairwise amino-acid identity of the tree datasets.** Each row is the percentage of pairs in bins
of 5% identity, on one scale throughout; the vertical line and the right-hand column give the median of the row. **a**, Every
pair of sequences within the *Flaviviridae* E (n = 236), E1 (n = 189) and E2 (n = 188) glycoprotein datasets, the sequences
of Fig. 2b and Supplementary Fig. 3. **b**, The 91 pairs among the 14 *Filoviridae* GP species representatives of Fig. 2a,
split by whether the pair crosses a genus. Identity is the fraction of alignment columns with a residue in both sequences at
which the residues are identical, from a MAFFT alignment of each dataset; the *Filoviridae* pairs are read off the alignment
of all 550 unique sequences. Over all 550, the median is 93% within a species, 59% between species of one genus and 31%
between genera; these pairs are not shown because the genera are sampled very unevenly, 90% of between-genus pairs being
*Orthoebolavirus* against *Orthomarburgvirus*.

**Supplementary Fig. 7 | Runtime vs sequence length.** Runtime of ESM3Di and ProstT5 on one A100 GPU for 512 sequences
per length bin: 32 Swiss-Prot viral proteins of about 100 to 1,000 residues per bin, each repeated 16 times. Mean ± s.d.
over five repetitions after a warm-up run. Filled markers: end to end, through a CLI call including model loading. Open markers: inference only, with the
model already loaded (measured for ESM3Di only), as in Fig. 1d.

**Supplementary Table 1 | *Filoviridae* glycoprotein (GP) accessions.** The 2,869 NCBI protein accessions with their species; the accession
representing each of the 550 unique sequences in the trees; whether that sequence is in the trees without X-rich sequences (at most 5% X, Supplementary Fig. 2); and the
accessions used as species representatives in Fig. 2a.

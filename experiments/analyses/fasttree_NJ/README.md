# FastTree NJ + NNI for protein alignments

## Research question and scope

This workflow explores a relatively inexpensive route from structure-aware
protein alignments to phylogenetic trees. FoldMason uses the amino acid and 3Di
tracks in a FoldSeek database to generate an alignment; FastTree then builds
a tree from the **amino acid alignment**, not directly from the 3Di strings.
The input database may contain structure-derived or model-predicted 3Di.

The archived scripts implement and document the method. No biological
benchmark results or completed tree outputs are tracked in this directory:
the presence of the workflow is not evidence that it outperforms another
alignment or tree-inference method.

## What was implemented

The experiment combines FoldMason fast-mode alignment without extra
refinement with FastTree's approximate neighbor-joining and minimum-evolution
NNI search. It deliberately omits maximum-likelihood refinement and support
estimation to reduce computation. Software versions, download checksums, and
compiler settings are recorded in [SOURCE.txt](SOURCE.txt); a concise
methods description is available in [methods.txt](methods.txt).

## Reproducing the workflow

The supported inference CLI now exposes this recipe without depending on
these archived shell scripts. Both repository Conda environment specs install
the upstream tools. See the
[inference commands](../../../README.md#foldmason-alignment-and-fasttree-inference)
for `foldmason`, `fasttree`, and `foldmason-fasttree` usage.
The original source-build procedure below remains available for reproducing
the archived native-CPU build.

Activate the repository Conda environment, which installs `foldmason` and
`FastTreeMP` on `PATH`. The scripts use those binaries by default, just like
the inference CLI:

```bash
cd experiments/analyses/fasttree_NJ
./run_fasttree_nj_nni.sh alignment.fasta tree.nwk 8
# Compressed input also works:
./run_fasttree_nj_nni.sh alignment.fasta.gz tree.nwk 16
tail -f tree.nwk.log
```

For the historical native-CPU build instead, run `./build_fasttree.sh` and
set `FASTTREE_BIN="$PWD/FastTreeMP"` when invoking either helper.
The build script downloads a pinned official source and checks its SHA256.
Dependencies: Bash, GCC with OpenMP, curl, sha256sum, gzip and standard GNU tools.
Rebuild on a different CPU because the binary uses `-march=native`.
FastTree uses its default double precision.

The helper runs `FastTreeMP -fastest -noml -spr 0 -nosupport`:

- `-fastest` accelerates heuristic neighbor joining and reduces memory use.
- `-noml` disables maximum-likelihood refinement, including ML branch lengths.
- `-spr 0` skips subtree-prune-regraft moves, leaving NJ followed by minimum-evolution NNI.
- `-nosupport` skips local support calculations.

NNI retains the default limit of approximately `4*log2(N)` rounds for N unique
sequences, with early stopping. Output is an unrooted Newick tree with
distance-based branch lengths and no support values. Protein distances use
FastTree's BLOSUM45 matrix; ML model flags such as `-lg` are unnecessary.

Input must be an amino acid alignment in FASTA or interleaved PHYLIP format,
with equal sequence lengths and unique identifiers. In FASTA, only the first
whitespace-delimited word of each header is used. Avoid `: , ( )` in identifiers.
3Di alignments share the amino acid alphabet but require an appropriate distance
model; this helper expects amino acids.

Output defaults to `<input>.nj_nni.nwk` (removing `.gz` if present). Diagnostics
are saved to `<tree>.log`. Existing outputs are refused, and failed runs do not
publish a partial tree. Threads default to `OMP_NUM_THREADS` if set, otherwise 8;
the third argument overrides this. NJ benefits from threads, but minimum-evolution
NNI is serial. OpenMP runs can produce slightly different topologies across runs.

## Foldseek database to tree

With the Conda environment active, run:

```bash
./run_foldmason_fasttree.sh /path/to/foldseek_db results/my_run 8
```

Pass the database **prefix**, with its amino acid, `_ss` (3Di), and `_h` (headers)
databases and their `.index`/`.dbtype` files alongside it. The output directory
must be new. No separate Foldseek executable is needed.

The pipeline runs `structuremsa --fast 1 --refine-iters 0` then passes
**`alignment_aa.fa`** to the FastTree helper. FoldMason fast mode disables residue
neighborhood scoring and works with predicted-3Di databases without C-alpha
coordinates. Databases with coordinates also work. Alignment refinement and
structure-scoring reports are omitted. FoldMason's initial all-versus-all
alignments can be expensive for very large databases; `-fastest` only affects
the subsequent FastTree analysis.

The run directory contains:

- `tree.nwk` and `tree.nwk.log`: final NJ + NNI tree and diagnostics.
- `alignment_aa.fa` and `alignment_3di.fa`: amino acid and 3Di alignments.
- `alignment.nw`: FoldMason's guide tree.
- `foldmason.log` and `run.txt`: alignment diagnostics and run settings.

Follow `foldmason.log`, then `tree.nwk.log`, with `tail -f` for progress. Failed
runs retain diagnostics; choose a new directory when retrying.

To use different installations (including the original local builds):

```bash
FOLDMASON_BIN=/path/to/foldmason FASTTREE_BIN=/path/to/FastTreeMP \
  ./run_foldmason_fasttree.sh db results/run2 8
```

`./setup_foldmason.sh` still installs the pinned Linux x86_64 AVX2 release
locally. To select it, set `FOLDMASON_BIN="$PWD/tools/foldmason/bin/foldmason"`.
Explicit overrides take precedence over `PATH`; an invalid override fails
before alignment starts rather than falling back to a different installation.

Source provenance and checksums are in [SOURCE.txt](SOURCE.txt). Setup downloads
FastTree source with its GPL v2-or-later notice and FoldMason with its upstream
README and license. Binaries, downloaded source, tools and analysis outputs
remain local and are ignored by Git. A methods paragraph is in [methods.txt](methods.txt).

Official documentation:

- https://morgannprice.github.io/fasttree/
- https://github.com/steineggerlab/foldmason

## Biological interpretation and limitations

- The output tree is unrooted and has no bootstrap or local-support values;
  it does not by itself establish ancestral direction or strongly supported
  relationships.
- Branch lengths are distance-based, not maximum-likelihood estimates.
  Tree differences can reflect alignment changes as well as model predictions.
- Predicted 3Di is a model-derived proxy for structure. Evaluate alignment and
  phylogenetic accuracy separately against appropriate reference data.
- If using historical `esm3di_from_mlm` model outputs, the old 3Di trainer did
  not transfer the intended MLM backbone weights. Such trees cannot demonstrate
  a benefit from MLM adaptation; that requires the separately trained fixed
  checkpoints and a controlled comparison.
- FastTree's protein distance model is not calibrated for a direct 3Di
  character alignment. The pipeline intentionally supplies `alignment_aa.fa`.

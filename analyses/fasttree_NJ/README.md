# FastTree NJ + NNI for protein alignments

Build FastTreeMP 2.2.0 with GCC, OpenMP and native CPU optimizations:

```bash
cd fasttree_NJ
./build_fasttree.sh
./run_fasttree_nj_nni.sh alignment.fasta tree.nwk 8
# Compressed input also works:
./run_fasttree_nj_nni.sh alignment.fasta.gz tree.nwk 16
tail -f tree.nwk.log
```

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

Install the pinned FoldMason Linux x86_64 AVX2 release, then run:

```bash
./setup_foldmason.sh
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

To use a different FoldMason installation (including other platforms):

```bash
FOLDMASON_BIN=/path/to/foldmason ./run_foldmason_fasttree.sh db results/run2 8
```

Source provenance and checksums are in [SOURCE.txt](SOURCE.txt). Setup downloads
FastTree source with its GPL v2-or-later notice and FoldMason with its upstream
README and license. Binaries, downloaded source, tools and analysis outputs
remain local and are ignored by Git. A methods paragraph is in [methods.txt](methods.txt).

Official documentation:

- https://morgannprice.github.io/fasttree/
- https://github.com/steineggerlab/foldmason

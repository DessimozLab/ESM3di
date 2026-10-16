#!/usr/bin/env bash
# Foldseek database -> FoldMason AA alignment -> FastTree NJ + NNI.
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: run_foldmason_fasttree.sh FOLDSEEK_DB OUTPUT_DIR [THREADS]

Input is a database prefix, not its .dbtype/.index file or a directory.
Requires amino acid, 3Di (_ss), and header (_h) databases.
OUTPUT_DIR must not exist. Alignments and logs are retained there.
Threads default to OMP_NUM_THREADS if set, otherwise 8.
FoldMason: structuremsa --fast 1 --refine-iters 0
FastTree: -fastest -noml -spr 0 -nosupport
Final tree: OUTPUT_DIR/tree.nwk
Override FoldMason with FOLDMASON_BIN=/path/to/foldmason.
Override FastTreeMP with FASTTREE_BIN=/path/to/FastTreeMP.
Defaults use foldmason and FastTreeMP from the active environment's PATH.
EOF
}
if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# < 2 || $# > 3 )); then usage >&2; exit 2; fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
db=$1
out=$2
threads=${3:-${OMP_NUM_THREADS:-8}}
foldmason=${FOLDMASON_BIN:-foldmason}
fasttree=${FASTTREE_BIN:-FastTreeMP}
if [[ ! $threads =~ ^[1-9][0-9]*$ ]]; then
    echo 'THREADS must be a positive integer.' >&2; exit 2
fi
if ! command -v -- "$foldmason" > /dev/null; then
    echo "Cannot find FoldMason executable: $foldmason; activate the Conda environment or set FOLDMASON_BIN." >&2; exit 2
fi
if ! command -v -- "$fasttree" > /dev/null; then
    echo "Cannot find FastTree executable: $fasttree; activate the Conda environment or set FASTTREE_BIN." >&2; exit 2
fi
if [[ ! -x $script_dir/run_fasttree_nj_nni.sh ]]; then
    echo "Missing executable helper: $script_dir/run_fasttree_nj_nni.sh" >&2; exit 2
fi
foldmason=$(command -v -- "$foldmason")
fasttree=$(command -v -- "$fasttree")
export FASTTREE_BIN="$fasttree"
for component in "$db" "${db}_ss" "${db}_h"; do
    for suffix in .dbtype .index; do
        if [[ ! -r ${component}${suffix} ]]; then
            echo "Missing database component: ${component}${suffix}" >&2; exit 2
        fi
    done
    if [[ ! -r $component && ! -r ${component}.0 ]]; then
        echo "Missing database data: $component (or ${component}.0)" >&2; exit 2
    fi
done
db=$(realpath -m -- "$db")
mkdir -p -- "$(dirname -- "$out")"
if ! mkdir -- "$out"; then
    echo 'Choose a new OUTPUT_DIR; existing runs are never overwritten.' >&2; exit 2
fi
out=$(cd -- "$out" && pwd)
{
    printf 'Database: %s\nThreads: %s\nFoldMason: %s\nFastTree: %s\n' "$db" "$threads" "$foldmason" "$fasttree"
    "$foldmason" version
    printf 'FoldMason options: structuremsa --fast 1 --refine-iters 0\n'
    printf 'FastTree options: -fastest -noml -spr 0 -nosupport\n'
} > "$out/run.txt"
echo "Aligning with FoldMason. Progress: $out/foldmason.log" >&2
if ! "$foldmason" structuremsa "$db" "$out/alignment" \
        --threads "$threads" --fast 1 --refine-iters 0 \
        > "$out/foldmason.log" 2>&1; then
    echo "FoldMason failed; see $out/foldmason.log" >&2; exit 1
fi
if [[ ! -s $out/alignment_aa.fa ]]; then
    echo "Missing amino acid alignment; see $out/foldmason.log" >&2; exit 1
fi
"$script_dir/run_fasttree_nj_nni.sh" "$out/alignment_aa.fa" "$out/tree.nwk" "$threads"
echo "Pipeline complete: $out/tree.nwk" >&2

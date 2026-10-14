#!/usr/bin/env bash
# Protein alignment -> heuristic NJ -> minimum-evolution NNI; no ML.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_fasttree_nj_nni.sh ALIGNMENT [TREE.nwk] [THREADS]

Input: aligned amino acid FASTA or interleaved PHYLIP, optionally gzip (.gz).
Default tree: ALIGNMENT.nj_nni.nwk (with .gz removed).
Default threads: OMP_NUM_THREADS if set, otherwise 8.
Progress/errors: TREE.nwk.log (follow with tail -f).
Existing output files are never overwritten.

Runs FastTreeMP -fastest -noml -spr 0 -nosupport.
NNI uses FastTree's default round limit (4*log2(unique sequences)).
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# < 1 || $# > 3 )); then usage >&2; exit 2; fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
input=$1
output=${2:-${input%.gz}.nj_nni.nwk}
threads=${3:-${OMP_NUM_THREADS:-8}}
log=${output}.log
if [[ ! $threads =~ ^[1-9][0-9]*$ ]]; then
    echo 'THREADS must be a positive integer.' >&2; exit 2
fi
if [[ ! -f $input || ! -r $input ]]; then
    echo "Cannot read alignment: $input" >&2; exit 2
fi
if [[ ! -x $script_dir/FastTreeMP ]]; then
    echo "Missing binary: $script_dir/FastTreeMP; run build_fasttree.sh." >&2; exit 2
fi
if [[ -e $output || -L $output || -e $log || -L $log ]]; then
    echo "Output already exists: $output or $log. Choose a new tree filename." >&2; exit 2
fi
export OMP_NUM_THREADS=$threads
tmp=$(mktemp -- "${output}.tmp.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(set -o noclobber; : > "$log")
echo "Running NJ + NNI with $threads threads. Progress: $log" >&2
args=(-fastest -noml -spr 0 -nosupport)
if [[ $input == *.gz ]]; then
    if gzip -dc -- "$input" | "$script_dir/FastTreeMP" "${args[@]}" > "$tmp" 2> "$log"; then
        :
    else
        echo "Analysis failed; see $log" >&2; exit 1
    fi
else
    if "$script_dir/FastTreeMP" "${args[@]}" < "$input" > "$tmp" 2> "$log"; then
        :
    else
        echo "Analysis failed; see $log" >&2; exit 1
    fi
fi
if [[ ! -s $tmp ]]; then echo "No tree produced; see $log" >&2; exit 1; fi
ln -- "$tmp" "$output"
echo "Tree written to $output" >&2

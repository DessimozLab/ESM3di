#!/usr/bin/env bash
# The two metagenomic trees of the paper, with the commands used for the figures. Files in ./rendered/ (not tracked).
# The figures themselves are in figures/ (see README.md); this script does not change them.
# Usage: ./render_figures.sh [main|supplementary|all]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p rendered
common=(--dpi 600 --layout unrooted --max-root-to-tip-factor 2.0 --rank class --max-tips 200000
        --purity 0.75 --max-colors 10 --tip-size 3 --linewidth 0.3 --no-show --quiet)

render_main() {     # Figure 2c: coloured by bacterial host class
    python3 massive_radial_tree.py \
        inputs/figure2c/IMGVR_PC_000000001_foldmason_3di.dedup.fa.nj_nni.nwk.gz \
        inputs/figure2c/IMGVR_PC_000000001_UViG.tsv.gz \
        --output rendered/IMGVR_PC_000000001.foldmason.png \
        --taxonomy-column host_taxonomy --z-order largest-front "${common[@]}"
}

render_supplementary() {     # Supplementary Fig. 7: coloured by protein cluster
    python3 massive_radial_tree.py \
        inputs/supplementary_figure7/imgvr_pc_000377566_foldmason.nwk.gz \
        inputs/supplementary_figure7/imgvr_pc_000377566_clu.tsv.gz --cluster-origin \
        --output rendered/imgvr_pc_000377566.foldmason.png \
        --z-order smallest-front "${common[@]}"
}

case ${1:-all} in
    main) render_main ;;
    supplementary) render_supplementary ;;
    all) render_main; render_supplementary ;;
    *) echo "Usage: $0 [main|supplementary|all]" >&2; exit 2 ;;
esac

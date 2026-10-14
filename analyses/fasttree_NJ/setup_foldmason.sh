#!/usr/bin/env bash
# Install the same official Linux x86_64 AVX2 release used for this pipeline.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]] || ! grep -qw avx2 /proc/cpuinfo; then
    echo 'This bundled release requires Linux x86_64 with AVX2. Install FoldMason separately and set FOLDMASON_BIN.' >&2
    exit 2
fi
if [[ -e tools/foldmason ]]; then
    echo 'tools/foldmason already exists; use it or move it before reinstalling.' >&2
    exit 2
fi
mkdir -p tools
scratch=$(mktemp -d tools/foldmason.download.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
curl -fL --retry 3 \
    https://github.com/steineggerlab/foldmason/releases/download/4-dd3c235/foldmason-linux-avx2.tar.gz \
    -o "$scratch/foldmason.tar.gz"
printf '%s  %s\n' 7e6f6bd264defda742882ec167eedaad1185c0a83a32646b1ec83fa6c5b86f05 \
    "$scratch/foldmason.tar.gz" | sha256sum -c -
tar -xzf "$scratch/foldmason.tar.gz" -C "$scratch"
mv -- "$scratch/foldmason" tools/foldmason
tools/foldmason/bin/foldmason version

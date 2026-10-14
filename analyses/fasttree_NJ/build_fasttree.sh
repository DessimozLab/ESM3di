#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source_sha=975202a6b74c9996af871404ff043bb2152edcbda539035662514bc12d1f3431
source_url=https://raw.githubusercontent.com/morgannprice/fasttree/a5a2723ea1e64faf3da7ea514521cfa348891add/FastTree.c
mkdir -p src
if [[ ! -f src/FastTree.c ]]; then
    download=$(mktemp src/FastTree.c.download.XXXXXX)
    trap 'rm -f -- "$download"' EXIT
    curl -fL --retry 3 "$source_url" -o "$download"
    printf '%s  %s\n' "$source_sha" "$download" | sha256sum -c -
    mv -- "$download" src/FastTree.c
fi
printf '%s  %s\n' "$source_sha" src/FastTree.c | sha256sum -c -
gcc -DOPENMP -O3 -fopenmp -fopenmp-simd -funsafe-math-optimizations \
    -march=native -o FastTreeMP src/FastTree.c -lm
echo 'Built FastTreeMP 2.2.0 for this machine.'

#!/usr/bin/env bash
set -eu

output=${1:?output path required}
ref=${2:-HEAD}
vsinclude=${VSINCLUDE:?VSINCLUDE must point to VapourSynth headers}
cxx=${CXX:-g++}
cxxflags=${CXXFLAGS:--O3 -std=c++17 -fPIC -Wall -Wextra -mavx2 -mfma}
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

git archive "$ref" src | tar -x -C "$tmp"
sources=(pipeline.cpp cuda_backend.cpp workspace.cpp maps.cpp kernels.cpp recon.cpp algos.cpp bm.cpp downsample.cpp plugin.cpp)
paths=()
for source in "${sources[@]}"; do
    paths+=("$tmp/src/$source")
done
"$cxx" -DLGCR_VERSION_MAJOR=1 -DLGCR_VERSION_MINOR=0 \
    -DLGCR_ENABLE_CUDA=0 -DLGCR_SUFFIX='"_baseline"' $cxxflags \
    -I"$vsinclude" -shared -o "$output" "${paths[@]}"

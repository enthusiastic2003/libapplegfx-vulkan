#!/bin/sh
# Build rust/lagfx-m2v (the metal2vulkan bridge) as a static library for meson.
#   build-m2v.sh <Cargo.toml> <cargo target dir> <output .a>
# Three jobs: the translator is large and the VM shares this machine's memory.
set -e
CARGO_TARGET_DIR="$2" cargo build --quiet --release -j "${LAGFX_CARGO_JOBS:-3}" --manifest-path "$1"
cp "$2/release/liblagfx_m2v.a" "$3"

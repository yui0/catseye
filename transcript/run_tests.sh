#!/bin/sh
# Regression runner.  C tests need only a compiler (add -DCATS_GPU when the EGL /
# GLESv2 development files exist, so the GPU test runs); the Python tests need
# numpy and are skipped when it is missing.
set -eu
cd "$(dirname "$0")"
compiler=${CC:-gcc}
work=$(mktemp -d)
# a few tests write their scratch output into the current directory
trap 'rm -rf "$work"; rm -f test_features.cfb test_native_graph.onnx test_native_stage3_graph.onnx' EXIT HUP INT TERM
extra=
if pkg-config --exists egl glesv2 2>/dev/null; then extra="-fopenmp -DCATS_GPU $(pkg-config --cflags --libs egl glesv2)"; fi
count=0
for src in test/test_*.c; do
  [ -e "$src" ] || { echo 'SKIP: no C tests in transcript/test'; break; }
  name=$(basename "${src%.c}")
  # shellcheck disable=SC2086
  "$compiler" -O"${CATS_TEST_OPT:-2}" -std=gnu11 -I. -o "$work/$name" "$src" -lm $extra
  "$work/$name" >/dev/null
  count=$((count + 1))
done
printf 'PASS: %s C tests\n' "$count"
if python3 -c 'import numpy' 2>/dev/null; then
  pycount=0
  for src in test/test_*.py; do
    [ -e "$src" ] || break
    PYTHONDONTWRITEBYTECODE=1 python3 "$src" >/dev/null 2>&1 || { echo "FAIL: $src"; exit 1; }
    pycount=$((pycount + 1))
  done
  printf 'PASS: %s Python tests\n' "$pycount"
else
  echo 'SKIP: Python tests (numpy not installed)'
fi

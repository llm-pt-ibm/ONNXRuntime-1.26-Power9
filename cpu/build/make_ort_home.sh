#!/bin/bash
# Assemble an ORT_HOME for onnxruntime-genai from our ppc64le ORT build.
#
# onnxruntime-genai expects the layout of the official release tarball:
#   $ORT_HOME/include/*.h   (flat - NOT the source tree's nested layout)
#   $ORT_HOME/lib/libonnxruntime.so*
set -euo pipefail

SRC=/root/onnx/onnxruntime
BUILD=/root/onnx/build/ort-cpu/Release
ORT_HOME=/root/onnx/ort-home-cpu

rm -rf "$ORT_HOME"
mkdir -p "$ORT_HOME/include" "$ORT_HOME/lib"

# Public C/C++ API headers, flattened.
cp "$SRC"/include/onnxruntime/core/session/*.h              "$ORT_HOME/include/"
cp "$SRC"/include/onnxruntime/core/framework/provider_options.h "$ORT_HOME/include/" 2>/dev/null || true
for p in cpu cuda; do
  cp "$SRC"/include/onnxruntime/core/providers/$p/*.h "$ORT_HOME/include/" 2>/dev/null || true
done

# Shared library + soname symlinks.
cp -P "$BUILD"/libonnxruntime.so* "$ORT_HOME/lib/"

echo "=== $ORT_HOME ==="
ls "$ORT_HOME/include/" | head -30
echo "---"
ls -la "$ORT_HOME/lib/"

#!/bin/bash
# Monta um ORT_HOME CUDA para o onnxruntime-genai a partir do nosso build ppc64le.
#
# Igual ao make_ort_home.sh da fase CPU, mas o /lib precisa das libs de provider:
#   libonnxruntime_providers_shared.so  (dispatcher que carrega EPs em runtime)
#   libonnxruntime_providers_cuda.so    (o CUDA EP em si)
# Sem elas o genai linka mas o CUDAExecutionProvider nao aparece em runtime.
set -euo pipefail

SRC=/root/onnx/onnxruntime
BUILD=/root/onnx/build/ort-cuda/Release
ORT_HOME=/root/onnx/ort-home-cuda

rm -rf "$ORT_HOME"
mkdir -p "$ORT_HOME/include" "$ORT_HOME/lib"

# Headers publicos da API C/C++, achatados.
cp "$SRC"/include/onnxruntime/core/session/*.h              "$ORT_HOME/include/"
cp "$SRC"/include/onnxruntime/core/framework/provider_options.h "$ORT_HOME/include/" 2>/dev/null || true
for p in cpu cuda; do
  cp "$SRC"/include/onnxruntime/core/providers/$p/*.h "$ORT_HOME/include/" 2>/dev/null || true
done

# Shared library + symlinks de soname + providers.
cp -P "$BUILD"/libonnxruntime.so* "$ORT_HOME/lib/"
cp -P "$BUILD"/libonnxruntime_providers_shared.so "$ORT_HOME/lib/" 2>/dev/null || true
cp -P "$BUILD"/libonnxruntime_providers_cuda.so   "$ORT_HOME/lib/" 2>/dev/null || true

echo "=== $ORT_HOME/include ==="
ls "$ORT_HOME/include/" | head -30
echo "=== $ORT_HOME/lib ==="
ls -la "$ORT_HOME/lib/"

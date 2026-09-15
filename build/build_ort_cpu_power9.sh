#!/bin/bash
# Build ONNX Runtime 1.26.0 (CPU) for ppc64le / POWER9
#
# Toolchain: conda-forge GCC 13.3 (powerpc64le-conda-linux-gnu).
#   - System GCC 8.5 is unusable (ICEs on vectorized C - see known-issues).
#   - conda clang 17 is ALSO unusable here: it picks up conda's libstdc++ 16
#     headers, which use __builtin_popcountg (a GCC 14+ / clang 19+ builtin)
#     -> "use of undeclared identifier '__builtin_popcountg'" in <atomic>.
#   - GCC 13.3 is self-consistent and stays nvcc-12.4 compatible for the
#     later CUDA stage.
set -x

ENV=/root/miniforge3/envs/onnx_build
SRC=/root/onnx/onnxruntime
BUILD=/root/onnx/build/ort-cpu

export PATH="$ENV/bin:$PATH"
export CC="$ENV/bin/powerpc64le-conda-linux-gnu-gcc"
export CXX="$ENV/bin/powerpc64le-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$ENV/lib:$LD_LIBRARY_PATH"

cd "$SRC" || exit 1

python tools/ci_build/build.py \
  --build_dir "$BUILD" \
  --config Release \
  --parallel 32 \
  --skip_submodule_sync \
  --allow_running_as_root \
  --compile_no_warning_as_error \
  --build_shared_lib \
  --build_wheel \
  --skip_tests \
  --cmake_extra_defines \
      onnxruntime_BUILD_UNIT_TESTS=OFF \
      CMAKE_C_COMPILER="$CC" \
      CMAKE_CXX_COMPILER="$CXX"

echo "=== EXIT CODE: $? ==="
ls -la "$BUILD/Release/libonnxruntime.so"* 2>&1
ls -la "$BUILD/Release/dist/" 2>&1

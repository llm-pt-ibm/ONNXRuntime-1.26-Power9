#!/bin/bash
# Build ONNX Runtime 1.26.0 com CUDA EP para ppc64le / POWER9 / Tesla V100 (sm_70)
#
# Continuacao de build_ort_cpu_power9.sh. Mesmo toolchain (conda GCC 13.3), porque
# o nvcc 12.4 barra em "__GNUC__ > 13" -- 13.3 passa, 14 nao passaria.
#
# Atencoes: na V100 (sm_70) o ORT 1.26 NAO desliga sozinho o flash attention por
# arquitetura; ele vem ON por default junto com USE_CUDA e exige sm_80. Tem que ser
# desligado a mao, senao o build quebra nos kernels cutlass.
#   FLASH_ATTENTION      OFF  <- exige sm_80
#   LEAN_ATTENTION       OFF  <- exige sm_8x/sm_90 (ja e default OFF)
#   FPA_INTB_GEMM        OFF  <- cutlass mixed-dtype, sm_80+ (ja e default OFF)
#   MEMORY_EFFICIENT_ATT ON   <- tem fmha_sm70.cu e dispatch "sm >= 70";
#                               e o que faz o GroupQueryAttention do genai rodar aqui
#
# O build e feito em duas fases porque o abseil precisa ser corrigido DEPOIS do
# configure (que e quem popula _deps/) e ANTES da compilacao. Ver patch_abseil_nvcc.sh.
set -x

ENV=/root/miniforge3/envs/onnx_build
SRC=/root/onnx/onnxruntime
BUILD=/root/onnx/build/ort-cuda
CUDA=/usr/local/cuda-12.4

export PATH="$ENV/bin:$CUDA/bin:$PATH"
export CC="$ENV/bin/powerpc64le-conda-linux-gnu-gcc"
export CXX="$ENV/bin/powerpc64le-conda-linux-gnu-g++"
export CUDA_HOME="$CUDA"
export CUDNN_HOME="$CUDA"
# NAO colocar /usr/local/cuda-12.2/compat aqui: a libcuda 535 de la conflita com o
# driver 550 do host e o cuInit passa a devolver 803 (SYSTEM_DRIVER_MISMATCH).
export LD_LIBRARY_PATH="$ENV/lib:$CUDA/lib64:$LD_LIBRARY_PATH"

COMMON_ARGS=(
  --build_dir "$BUILD"
  --config Release
  --parallel 40
  --nvcc_threads 2
  --skip_submodule_sync
  --allow_running_as_root
  --compile_no_warning_as_error
  --build_shared_lib
  --build_wheel
  --skip_tests
  --use_cuda
  --cuda_home "$CUDA"
  --cudnn_home "$CUDA"
  --cuda_version 12.4
  --cmake_extra_defines
      onnxruntime_BUILD_UNIT_TESTS=OFF
      CMAKE_CUDA_ARCHITECTURES=70
      CMAKE_C_COMPILER="$CC"
      CMAKE_CXX_COMPILER="$CXX"
      CMAKE_CUDA_HOST_COMPILER="$CXX"
      onnxruntime_USE_FLASH_ATTENTION=OFF
      onnxruntime_USE_LEAN_ATTENTION=OFF
      onnxruntime_USE_FPA_INTB_GEMM=OFF
      onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=ON
)

cd "$SRC" || exit 1

# Fase 0: exporta Einsum::DeviceCompute (tem que ser ANTES do configure, porque o
# .lds e gerado a partir do symbols.txt). Sem isso o provider .so nao carrega e a
# sessao cai para CPU silenciosamente.
bash /root/onnx/patch_ort_einsum_symbol.sh "$SRC" || exit 1

# Fase 1: configure (popula _deps/abseil_cpp-src)
python tools/ci_build/build.py "${COMMON_ARGS[@]}" --update || exit 1

# Fase 2: corrige o abseil para o nvcc
bash /root/onnx/patch_abseil_nvcc.sh \
  "$BUILD/Release/_deps/abseil_cpp-src/absl/container/internal/common.h" || exit 1

# Fase 3: remove artefatos que nao sao reconstruidos sozinhos quando so os
# version scripts mudam (o cmake nao os trata como dependencia de link), e o
# wheel acaba reempacotando copias velhas.
rm -f "$BUILD/Release/onnxruntime_pybind11_state.so" \
      "$BUILD/Release/libonnxruntime.so"* \
      "$BUILD/Release/dist/"*.whl

# Fase 4: compila
python tools/ci_build/build.py "${COMMON_ARGS[@]}" --build

echo "=== EXIT CODE: $? ==="
ls -la "$BUILD/Release/libonnxruntime.so"* 2>&1
ls -la "$BUILD/Release/libonnxruntime_providers_cuda.so" 2>&1
ls -la "$BUILD/Release/dist/" 2>&1

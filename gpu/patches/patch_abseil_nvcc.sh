#!/bin/bash
# Corrige a incompatibilidade abseil 20250814 x nvcc 12.4 que quebra TODOS os .cu
# do onnxruntime_providers_cuda (32 arquivos; 0 erros nos .cc).
#
# Causa: em raw_hash_map.h o macro ABSL_INTERNAL_X usa "IfRRef<int KQual>::AddPtr<K>".
# Com KQual = "const &" cai no template primario, onde AddPtr<K> COLAPSA para K --
# um parametro de template nu. O frontend EDG do nvcc reescreve o intermediario
# inserindo um "typename" ali, e o GCC entao rejeita:
#   error: using template type parameter [...]IfRRef<const int&>::AddPtr<K> after typename
# (prefixar "typename" a um parametro de template e ilegal).
#
# Fix: fazer AddPtr nunca colapsar para um nome nu, preservando o mesmo tipo.
#   using AddPtr = Other;   ->   using AddPtr = typename std::enable_if<true, Other>::type;
# common.h ja inclui <type_traits>. Validado com reprodutor de 3 linhas:
#   antes  -> nvcc 22 erros / g++ 0 erros
#   depois -> nvcc 0 erros (gera .o) / g++ 0 erros
#
# Uso: patch_abseil_nvcc.sh <caminho para absl/container/internal/common.h>
set -euo pipefail
C="$1"
if grep -q "enable_if<true, Other>" "$C"; then
  echo "patch_abseil_nvcc: ja aplicado em $C"
  exit 0
fi
cp -n "$C" "$C.orig"
sed -i \
  -e "s|  using AddPtr = Other;|  using AddPtr = typename std::enable_if<true, Other>::type;|" \
  -e "s|  using AddPtr = Other\*;|  using AddPtr = typename std::enable_if<true, Other>::type*;|" \
  "$C"
grep -q "enable_if<true, Other>" "$C" || { echo "patch_abseil_nvcc: FALHOU em $C"; exit 1; }
echo "patch_abseil_nvcc: aplicado em $C"

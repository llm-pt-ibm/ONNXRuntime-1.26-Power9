#!/bin/bash
# Corrige o CUDA EP que nao carrega: "undefined symbol: onnxruntime::Einsum::DeviceCompute".
#
# SINTOMA (o pior tipo): a sessao CUDA cai para CPU em SILENCIO. O
# CUDAExecutionProvider APARECE em ort.get_available_providers() -- o dlopen e que
# falha -- e o ORT so loga um aviso generico "Require cuDNN 9.* and CUDA 12.*",
# mandando voce investigar a instalacao de CUDA, que nao tem nada a ver.
# Sem checar sess.get_providers() depois de criar a sessao, isso passa por "funcionou".
#
# CAUSA: provider_bridge_provider.cc e compilado DENTRO do provider .so e define
# Einsum::Compute. Pela regra da key function (a vtable e emitida onde a primeira
# virtual nao-inline e definida), isso emite ali a vtable de onnxruntime::Einsum,
# que precisa do endereco de TODAS as virtuais -- inclusive DeviceCompute, definida
# so do lado CPU, em libonnxruntime_providers.a.
#
# E codigo morto: o Einsum do CUDA e onnxruntime::cuda::Einsum, que deriva de
# CudaKernel e NAO de onnxruntime::Einsum. Nenhum objeto do provider tem esse tipo.
# Deveria ser descartado por --gc-sections (que esta ativo, junto com
# -ffunction-sections/-fdata-sections), mas a vtable sobrevive aqui.
#
# POR QUE EXPORTAR NAO RESOLVE (duas tentativas que falharam antes desta):
#   1. Acrescentar o simbolo a core/providers/cuda/symbols.txt exporta do
#      libonnxruntime.so, mas o mesmo arquivo alimenta o generated_source.c -- um
#      arquivo C com um mapa nome->endereco -- e um nome C++ mangled nao e
#      identificador C valido: "'_ZNK11onnxruntime6Einsum...' undeclared".
#   2. Exportar tambem do onnxruntime_pybind11_state.so (que linka o ORT
#      estaticamente e e o host real no caminho Python) compila, mas nao adianta:
#      o Python carrega modulos de extensao com RTLD_LOCAL, entao os simbolos dele
#      nao entram no escopo global e o provider carregado depois nao os enxerga.
#
# FIX: definir o simbolo DENTRO do proprio provider. Resolve os dois hosts
# (libonnxruntime.so para C/C++ / genai, e o pybind para Python) e nao depende de
# escopo de linker. Como e inalcancavel, a implementacao falha alto se um dia for
# chamada -- nada de retornar OK em silencio.
set -euo pipefail
SRC="$1"   # raiz do repo onnxruntime
F="$SRC/onnxruntime/core/providers/shared_library/provider_bridge_provider.cc"

if grep -q "Einsum::DeviceCompute" "$F"; then
  echo "patch_ort_einsum_symbol: ja aplicado"
  exit 0
fi
cp -n "$F" "$F.orig"
python - "$F" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
anchor = "Status Einsum::Compute(OpKernelContext* context) const { return g_host_cpu.Einsum__Compute(this, context); }\n"
addition = anchor + """
// Definir Einsum::Compute acima faz o compilador emitir aqui a vtable de
// onnxruntime::Einsum (regra da key function), e a vtable precisa do endereco de
// DeviceCompute, que so existe do lado CPU. Sem esta definicao o provider .so fica
// com um simbolo indefinido e o dlopen falha -- a sessao CUDA cai para CPU em
// silencio, com um aviso enganoso sobre cuDNN/CUDA.
//
// Inalcancavel na pratica: o Einsum do CUDA e onnxruntime::cuda::Einsum, derivado
// de CudaKernel, nao desta classe. Falha alto caso a premissa mude.
Status Einsum::DeviceCompute(OpKernelContext*, const std::vector<const Tensor*>&,
                             AllocatorPtr, concurrency::ThreadPool*) const {
  ORT_NOT_IMPLEMENTED("Einsum::DeviceCompute is not available in a provider shared library");
}
"""
assert src.count(anchor) == 1, f"esperava 1 ancora, achei {src.count(anchor)}"
open(path, "w").write(src.replace(anchor, addition))
PYEOF
grep -q "Einsum::DeviceCompute" "$F" || { echo "patch_ort_einsum_symbol: FALHOU"; exit 1; }
echo "patch_ort_einsum_symbol: aplicado em $F"

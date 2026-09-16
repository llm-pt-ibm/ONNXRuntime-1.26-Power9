#!/bin/bash
# Progresso do build corrente. Uso: bash /root/onnx/status.sh [logfile]
LOG=${1:-/root/onnx/build_ort_cpu.log}

echo "===== $LOG ====="
echo "--- rodando? ---"
pgrep -af "build.py|ninja|cmake" | head -5 || echo "(nenhum processo de build ativo)"

echo
echo "--- ultimas linhas ---"
tail -15 "$LOG" 2>/dev/null

echo
echo "--- progresso ninja (ultimo [N/M]) ---"
grep -oE "\[[0-9]+/[0-9]+\]" "$LOG" 2>/dev/null | tail -1

echo
echo "--- erros ---"
grep -niE "error:|CMake Error|FAILED:|fatal error" "$LOG" 2>/dev/null | tail -15

echo
echo "--- artefatos ---"
ls -la /root/onnx/build/ort-cpu/Release/libonnxruntime.so* 2>/dev/null
ls -la /root/onnx/build/ort-cpu/Release/dist/*.whl 2>/dev/null

#!/usr/bin/env python
"""
Isolate the two candidate causes of slow LLM decoding on POWER9:

  (a) thread oversubscription  - 160 SMT threads on a tiny decode GEMM
  (b) missing int4 kernel      - MLAS registers QNBitGemmDispatch only for
                                 AVX2/AVX512/NEON/LASX, never for POWER

Compares plain fp32 MatMul (which DOES have a POWER9 MLAS kernel:
SgemmKernelPower.cpp) against MatMulNBits int4 (which does not), at the GEMM
shapes Phi-3-mini actually hits during decoding, across thread counts.

Usage: python bench_matmul.py
"""

import time

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

# Phi-3-mini: hidden 3072, intermediate 8192. Decoding is M=1.
SHAPES = [(1, 3072, 3072), (1, 3072, 8192), (64, 3072, 8192)]
THREADS = [1, 4, 8, 16, 20, 40, 80, 160]
BLOCK = 32
REPEATS = 30


def sess(model, nthreads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = nthreads
    so.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])


def matmul_fp32_model(m, k, n):
    rng = np.random.default_rng(0)
    b = rng.standard_normal((k, n), dtype=np.float32)
    init = helper.make_tensor("B", TensorProto.FLOAT, [k, n], b.flatten().tolist())
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        initializer=[init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def matmul_nbits_model(m, k, n, bits=4, block=BLOCK):
    """MatMulNBits (com.microsoft): B is int4, packed 2 values per byte."""
    rng = np.random.default_rng(1)
    n_blocks = (k + block - 1) // block
    packed = rng.integers(0, 256, size=(n, n_blocks, block // 2), dtype=np.uint8)
    scales = (rng.random((n * n_blocks,), dtype=np.float32) * 0.02 + 0.01)

    inits = [
        helper.make_tensor("B", TensorProto.UINT8, list(packed.shape),
                           packed.flatten().tobytes(), raw=True),
        helper.make_tensor("scales", TensorProto.FLOAT, [n * n_blocks],
                           scales.flatten().tolist()),
    ]
    node = helper.make_node(
        "MatMulNBits", ["A", "B", "scales"], ["Y"],
        domain="com.microsoft",
        K=k, N=n, bits=bits, block_size=block,
    )
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    return model


def bench(s, feeds):
    for _ in range(3):
        s.run(None, feeds)
    t0 = time.time()
    for _ in range(REPEATS):
        s.run(None, feeds)
    return (time.time() - t0) / REPEATS


def main():
    print(f"onnxruntime {ort.__version__}  |  repeticoes={REPEATS}")
    for m, k, n in SHAPES:
        a = np.random.default_rng(2).standard_normal((m, k), dtype=np.float32)
        flops = 2.0 * m * k * n
        print(f"\n### GEMM M={m} K={k} N={n}   ({flops / 1e6:.1f} MFLOP por chamada)")
        print(f"{'threads':>8}  {'fp32 ms':>9}  {'fp32 GFLOP/s':>13}   "
              f"{'int4 ms':>9}  {'int4 GFLOP/s':>13}  {'int4/fp32':>10}")
        print("-" * 78)

        mf = matmul_fp32_model(m, k, n)
        mn = matmul_nbits_model(m, k, n)

        for t in THREADS:
            tf = bench(sess(mf, t), {"A": a})
            try:
                tn = bench(sess(mn, t), {"A": a})
                ratio = f"{tn / tf:.1f}x"
                tn_s = f"{tn * 1e3:>9.2f}"
                gn = f"{flops / tn / 1e9:>13.1f}"
            except Exception as exc:
                tn_s, gn, ratio = "   ERRO", "         -", str(exc)[:10]
            print(f"{t:>8}  {tf * 1e3:>9.2f}  {flops / tf / 1e9:>13.1f}   "
                  f"{tn_s}  {gn}  {ratio:>10}", flush=True)


if __name__ == "__main__":
    main()

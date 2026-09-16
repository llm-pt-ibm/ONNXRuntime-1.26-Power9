#!/usr/bin/env python
"""
Correctness of MatMulNBits (int4) against a NumPy reference.

This is the test that guards the POWER SQNBitGemm kernel. It must pass both
before the kernel exists (ORT's ComputeBUnpacked fallback) and after (the MLAS
path), with identical results - that equality is the whole point.

Two code paths are covered deliberately:
  M == 1  -> SQ4BitGemmM1Kernel_CompFp32   (the decode hot path)
  M  > 1  -> SQ4BitBlkDequantBForSgemm_CompFp32 + the POWER sgemm kernel

Getting the nibble order or the tile layout wrong produces plausible-looking
but wrong numbers, so every case is checked elementwise.

Run: python test_matmulnbits_correctness.py
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

FAILURES = []
PASSES = []


def unpack_nibbles(packed, blk_len):
    """packed: (..., blk_len//2) uint8 -> (..., blk_len) uint8.

    Byte i holds value 2i in the low nibble and value 2i+1 in the high nibble.
    """
    lo = packed & 0x0F
    hi = packed >> 4
    out = np.empty(packed.shape[:-1] + (blk_len,), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def reference_matmul_nbits(a, packed, scales, zero_points, n, k, blk_len):
    """Dequantize B and multiply, in float64 to keep the reference clean."""
    n_blocks = (k + blk_len - 1) // blk_len

    vals = unpack_nibbles(packed, blk_len).astype(np.float64)   # (n, n_blocks, blk_len)
    sc = scales.reshape(n, n_blocks).astype(np.float64)

    if zero_points is None:
        zp = np.full((n, n_blocks), 8.0)
    else:
        zp_un = unpack_nibbles(zero_points.reshape(n, -1), 2 * zero_points.shape[-1])
        zp = zp_un[:, :n_blocks].astype(np.float64)

    deq = (vals - zp[:, :, None]) * sc[:, :, None]      # (n, n_blocks, blk_len)
    b = deq.reshape(n, n_blocks * blk_len)[:, :k].T      # (k, n)
    return a.astype(np.float64) @ b


def build_model(m, n, k, blk_len, packed, scales, zero_points):
    inits = [
        helper.make_tensor("B", TensorProto.UINT8, list(packed.shape),
                           packed.tobytes(), raw=True),
        helper.make_tensor("scales", TensorProto.FLOAT, [scales.size],
                           scales.flatten().tolist()),
    ]
    inputs = ["A", "B", "scales"]
    if zero_points is not None:
        inits.append(helper.make_tensor("zero_points", TensorProto.UINT8,
                                        list(zero_points.shape),
                                        zero_points.tobytes(), raw=True))
        inputs.append("zero_points")

    node = helper.make_node("MatMulNBits", inputs, ["Y"], domain="com.microsoft",
                            K=k, N=n, bits=4, block_size=blk_len)
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


def case(m, n, k, blk_len, with_zp, seed):
    rng = np.random.default_rng(seed)
    n_blocks = (k + blk_len - 1) // blk_len

    packed = rng.integers(0, 256, size=(n, n_blocks, blk_len // 2), dtype=np.uint8)
    scales = (rng.random((n * n_blocks,)) * 0.05 + 0.005).astype(np.float32)
    zero_points = None
    if with_zp:
        zp_bytes = (n_blocks + 1) // 2
        zero_points = rng.integers(0, 256, size=(n, zp_bytes), dtype=np.uint8)

    a = rng.standard_normal((m, k)).astype(np.float32)

    model = build_model(m, n, k, blk_len, packed, scales, zero_points)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"A": a})

    want = reference_matmul_nbits(a, packed, scales, zero_points, n, k, blk_len)

    name = f"M={m} N={n} K={k} blk={blk_len} zp={'sim' if with_zp else 'nao'}"
    # int4 dequant is exact; the only error is fp32 accumulation order.
    tol = 2e-3 * max(1.0, np.abs(want).max())
    diff = np.abs(got.astype(np.float64) - want).max()
    if diff > tol:
        idx = np.unravel_index(np.argmax(np.abs(got.astype(np.float64) - want)), want.shape)
        FAILURES.append(f"{name}: max|diff|={diff:.6g} > tol={tol:.6g} "
                        f"(got={got[idx]!r} want={want[idx]!r})")
    else:
        PASSES.append(f"{name}  (max|diff|={diff:.2e})")


def main():
    print(f"onnxruntime {ort.__version__} | {__import__('platform').machine()}")
    print("-" * 72)

    seed = 0
    for blk_len in (16, 32, 64, 128):
        for with_zp in (False, True):
            # M=1 exercises SQ4BitGemmM1Kernel_CompFp32
            case(1, 32, 256, blk_len, with_zp, seed); seed += 1
            # M>1 exercises SQ4BitBlkDequantBForSgemm_CompFp32 + sgemm
            case(4, 32, 256, blk_len, with_zp, seed); seed += 1

    # Shapes that stress tile edges: N not a multiple of 16, K not a multiple
    # of BlkLen, and a wide N spanning several 16-column tiles.
    case(1, 17, 96, 32, False, 100)
    case(3, 17, 96, 32, True, 101)
    case(1, 129, 512, 32, False, 102)
    case(8, 129, 512, 32, True, 103)
    case(1, 1, 32, 32, False, 104)
    case(2, 48, 160, 32, True, 105)
    # Realistic decode shape from Phi-3-mini.
    case(1, 3072, 3072, 32, False, 106)

    for p in PASSES:
        print(f"  ok    {p}")
    print("-" * 72)
    if FAILURES:
        print(f"FALHAS ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  FALHOU  {f}")
        print(f"\n{len(PASSES)} passaram, {len(FAILURES)} falharam")
        return 1
    print(f"{len(PASSES)} passaram, 0 falharam")
    return 0


if __name__ == "__main__":
    sys.exit(main())

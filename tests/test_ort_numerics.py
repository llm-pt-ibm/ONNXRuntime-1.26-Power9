#!/usr/bin/env python
"""
Numerical correctness battery for ONNX Runtime on ppc64le / POWER9.

Targets the failure classes that actually bite on Power (see the team's
known-issues catalog): int8 sign handling (plain `char` is unsigned on PPC),
float16/bfloat16 conversion paths, and the POWER9 VSX MLAS GEMM/quantize
kernels. Every case is checked against a NumPy reference.

Run: python test_ort_numerics.py
Exit code 0 = all passed.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

FAILURES = []
PASSES = []


def check(name, got, want, atol=1e-5, rtol=1e-5):
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        FAILURES.append(f"{name}: shape {got.shape} != esperado {want.shape}")
        return
    if not np.allclose(got, want, atol=atol, rtol=rtol):
        bad = np.argmax(np.abs(got.astype(np.float64) - want.astype(np.float64)))
        FAILURES.append(
            f"{name}: max |diff|={np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))):.6g} "
            f"(got[{bad}]={got.flat[bad]!r} want[{bad}]={want.flat[bad]!r})"
        )
        return
    PASSES.append(name)


def run(model, feeds):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def single_node(op, inputs, outputs, elem_types, shapes, **attrs):
    """Build a one-node model. elem_types/shapes cover inputs then outputs."""
    node = helper.make_node(op, inputs, outputs, **attrs)
    n_in = len(inputs)
    graph = helper.make_graph(
        [node],
        f"g_{op}",
        [helper.make_tensor_value_info(n, elem_types[i], shapes[i]) for i, n in enumerate(inputs)],
        [
            helper.make_tensor_value_info(n, elem_types[n_in + i], shapes[n_in + i])
            for i, n in enumerate(outputs)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------- float32 GEMM
def test_matmul_f32():
    rng = np.random.default_rng(0)
    for m, k, n in [(1, 1, 1), (7, 13, 5), (64, 64, 64), (128, 256, 64), (3, 512, 129)]:
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        model = single_node(
            "MatMul", ["A", "B"], ["Y"],
            [TensorProto.FLOAT] * 3, [[m, k], [k, n], [m, n]],
        )
        (y,) = run(model, {"A": a, "B": b})
        check(f"MatMul f32 {m}x{k}x{n}", y, a @ b, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- float64 GEMM
def test_matmul_f64():
    rng = np.random.default_rng(1)
    m, k, n = 33, 65, 17
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    model = single_node(
        "MatMul", ["A", "B"], ["Y"],
        [TensorProto.DOUBLE] * 3, [[m, k], [k, n], [m, n]],
    )
    (y,) = run(model, {"A": a, "B": b})
    check(f"MatMul f64 {m}x{k}x{n}", y, a @ b, atol=1e-12, rtol=1e-12)


# ------------------------------------------------- int8: the PPC danger zone
def test_int8_roundtrip():
    """`char` is unsigned on PowerPC - negative int8 must survive conversion."""
    vals = np.array([-128, -100, -4, -1, 0, 1, 4, 100, 127], dtype=np.int8)
    model = single_node(
        "Cast", ["X"], ["Y"],
        [TensorProto.INT8, TensorProto.FLOAT], [[len(vals)], [len(vals)]],
        to=TensorProto.FLOAT,
    )
    (y,) = run(model, {"X": vals})
    check("Cast int8->f32 (sinal)", y, vals.astype(np.float32))


def test_quantize_dequantize_s8():
    """Exercises the POWER9 VSX quantize kernel (QuantizePowerVSX.cpp)."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-8, 8, size=1024).astype(np.float32)
    # 0-d arrays, not numpy scalars: ORT's feed binding rejects the latter.
    scale = np.array(0.05, dtype=np.float32)
    zp = np.array(-3, dtype=np.int8)

    q_model = single_node(
        "QuantizeLinear", ["X", "S", "Z"], ["Y"],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT8, TensorProto.INT8],
        [[1024], [], [], [1024]],
    )
    (q,) = run(q_model, {"X": x, "S": scale, "Z": zp})
    ref_q = np.clip(np.rint(x / scale) + zp, -128, 127).astype(np.int8)
    check("QuantizeLinear int8", q.astype(np.int32), ref_q.astype(np.int32), atol=0, rtol=0)

    dq_model = single_node(
        "DequantizeLinear", ["X", "S", "Z"], ["Y"],
        [TensorProto.INT8, TensorProto.FLOAT, TensorProto.INT8, TensorProto.FLOAT],
        [[1024], [], [], [1024]],
    )
    (dq,) = run(dq_model, {"X": ref_q, "S": scale, "Z": zp})
    check("DequantizeLinear int8", dq, (ref_q.astype(np.float32) - zp) * scale, atol=1e-6)


def test_quantize_dequantize_u8():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, size=777).astype(np.float32)
    scale = np.array(0.04, dtype=np.float32)
    zp = np.array(128, dtype=np.uint8)
    q_model = single_node(
        "QuantizeLinear", ["X", "S", "Z"], ["Y"],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.UINT8, TensorProto.UINT8],
        [[777], [], [], [777]],
    )
    (q,) = run(q_model, {"X": x, "S": scale, "Z": zp})
    ref_q = np.clip(np.rint(x / scale) + zp, 0, 255).astype(np.uint8)
    check("QuantizeLinear uint8", q.astype(np.int32), ref_q.astype(np.int32), atol=0, rtol=0)


def test_matmul_integer():
    """Integer GEMM - the POWER10/POWER9 qgemm kernels."""
    rng = np.random.default_rng(4)
    m, k, n = 17, 96, 23
    a = rng.integers(-128, 128, size=(m, k)).astype(np.int8)
    b = rng.integers(-128, 128, size=(k, n)).astype(np.int8)
    model = single_node(
        "MatMulInteger", ["A", "B"], ["Y"],
        [TensorProto.INT8, TensorProto.INT8, TensorProto.INT32],
        [[m, k], [k, n], [m, n]],
    )
    (y,) = run(model, {"A": a, "B": b})
    check("MatMulInteger s8xs8", y, a.astype(np.int32) @ b.astype(np.int32), atol=0, rtol=0)


# ---------------------------------------------------------------- float16
def test_float16_matmul():
    rng = np.random.default_rng(5)
    m, k, n = 32, 64, 16
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    model = single_node(
        "MatMul", ["A", "B"], ["Y"],
        [TensorProto.FLOAT16] * 3, [[m, k], [k, n], [m, n]],
    )
    (y,) = run(model, {"A": a, "B": b})
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    check("MatMul f16", y.astype(np.float32), ref.astype(np.float32), atol=0.05, rtol=0.05)


def test_float16_edge_values():
    """f16 special values through a Cast - catches conversion-path bugs."""
    vals = np.array(
        [0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0, 6.1035156e-05, -6.1035156e-05,
         np.inf, -np.inf],
        dtype=np.float16,
    )
    model = single_node(
        "Cast", ["X"], ["Y"],
        [TensorProto.FLOAT16, TensorProto.FLOAT], [[len(vals)], [len(vals)]],
        to=TensorProto.FLOAT,
    )
    (y,) = run(model, {"X": vals})
    check("Cast f16->f32 (extremos)", y, vals.astype(np.float32))

    back = single_node(
        "Cast", ["X"], ["Y"],
        [TensorProto.FLOAT, TensorProto.FLOAT16], [[len(vals)], [len(vals)]],
        to=TensorProto.FLOAT16,
    )
    (y2,) = run(back, {"X": vals.astype(np.float32)})
    check("Cast f32->f16 (extremos)", y2.astype(np.float32), vals.astype(np.float32))


# ---------------------------------------------------------------- softmax/reduce
def test_softmax():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((8, 512)).astype(np.float32) * 10
    model = single_node(
        "Softmax", ["X"], ["Y"],
        [TensorProto.FLOAT] * 2, [[8, 512], [8, 512]], axis=-1,
    )
    (y,) = run(model, {"X": x})
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    check("Softmax f32", y, e / e.sum(axis=-1, keepdims=True), atol=1e-6)


def test_reduce_sum_large():
    """Long accumulation - shakes out VSX reduction bugs."""
    x = np.full((1, 100000), 0.1, dtype=np.float32)
    model = single_node(
        "ReduceSum", ["X"], ["Y"],
        [TensorProto.FLOAT] * 2, [[1, 100000], [1, 1]], keepdims=1,
    )
    (y,) = run(model, {"X": x})
    check("ReduceSum 1e5 elementos", y, np.array([[x.sum()]]), atol=1.0, rtol=1e-3)


# ---------------------------------------------------------------- endianness
def test_serialization_roundtrip():
    """Model bytes are little-endian; ppc64le is LE too, so this must match."""
    rng = np.random.default_rng(7)
    w = rng.standard_normal((4, 4)).astype(np.float32)
    init = helper.make_tensor("W", TensorProto.FLOAT, [4, 4], w.flatten().tolist())
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])],
        initializer=[init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    x = np.eye(4, dtype=np.float32)
    (y,) = run(model, {"X": x})
    check("Initializer serializado (endianness)", y, w, atol=1e-6)


def main():
    print(f"onnxruntime {ort.__version__}  |  providers: {ort.get_available_providers()}")
    print(f"numpy {np.__version__}  |  plataforma: {sys.platform} {__import__('platform').machine()}")
    print("-" * 70)

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{t.__name__}: EXCEÇÃO {type(exc).__name__}: {exc}")

    for name in PASSES:
        print(f"  ok    {name}")
    print("-" * 70)
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

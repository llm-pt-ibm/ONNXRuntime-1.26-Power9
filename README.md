# ONNX Runtime 1.26.0 for POWER9 (ppc64le) — with an int4 MLAS kernel

ONNX Runtime builds on ppc64le out of the box: MLAS has had POWER GEMM and
quantization kernels for years. What it did **not** have was a kernel for
`MatMulNBits` — the 4-bit blockwise-quantized matrix multiply that dominates
the runtime of a quantized LLM.

Without it `MlasIsQNBitGemmAvailable()` returned false on POWER and ONNX Runtime
silently fell back to `ComputeBUnpacked()`, which decompresses the entire weight
matrix on every call. Results were correct and roughly **45× slower than the
same GEMM in fp32**.

This repository contains that missing kernel, written in VSX, plus the build
scripts, tests and benchmarks used to develop and validate it.

## Results

Phi-3-mini 3.8B, int4 RTN block-32, decoding throughput (best thread count):

| Stage | tok/s | vs. stock |
|---|---|---|
| Stock ONNX Runtime, default threads | 0.13 | — |
| Stock ONNX Runtime, tuned threads | 0.51 | 3.9× |
| 1. Scalar kernel on the MLAS path | 2.66 | 20× |
| 2. VSX nibble unpacking | 7.30 | 56× |
| 3. Four columns at a time | 7.89 | 61× |
| 4. Vector accumulator across blocks | 9.01 | 69× |
| 5. Full 32-value block per load | **10.48** | **81×** |

Controlled A/B on Llama 3.2 1B Instruct (int4, acc-level-4) — same machine, same
model, same script, only the kernel differs. Wheel identity verified by md5
before each measurement:

| Threads | Without the kernel | With the kernel |
|---|---|---|
| 20 | 1.50 | 21.61 |
| 40 | 1.45 | 21.97 |
| **80** | 1.28 | **23.49** |
| 160 (ONNX Runtime default) | 0.32 | 9.65 |

`MatMulNBits` in isolation, M=1 K=3072 N=8192, peak over thread counts:
**3.2 → 67.9 GFLOP/s**. int4 went from 25–45× slower than fp32 to 1.1–1.8×.

> **Thread count matters as much as the kernel on this machine.** The box has
> 160 logical CPUs (2 sockets × 20 cores × SMT4) and ONNX Runtime uses all of
> them by default, which costs 2.4× on the small per-token GEMMs of decoding.
> Set `intra_op_num_threads` between 40 and 80.

## What the kernel does

`MatMulNBits` weights are stored compressed: 4-bit integers packed two per byte,
plus one scale (and optional zero point) per block of 32 values. Reconstructing
a weight is `(nibble - zero_point) * scale`.

Only two entry points are strictly required — `MlasIsQNBitGemmAvailable()`
checks exactly these for the `SQNBIT_CompFp32` variant:

| Function | Case | Strategy |
|---|---|---|
| `SQ4BitGemmM1Kernel_CompFp32` | M = 1 (decoding) | unpack and dot-product in one fused pass |
| `SQ4BitBlkDequantBForSgemm_CompFp32` | M > 1 (prompt) | dequantize a tile, hand it to the existing POWER sgemm kernel |

Packing must also be registered: `MatMulNBits` passes `packed_b_` as
`PackedQuantBData` unconditionally, so a `PackQuantBDataSize` of 0 leaves the
kernel with a null pointer. Identity packing (a `memcpy`) is used — the kernels
read B in its native layout.

### The core trick

In the native layout the even-indexed value lives in the low nibble of each byte
and the odd-indexed one in the high nibble. So separating the two halves and
interleaving them recovers the correct order for free:

```c
bytes = vec_xl(0, src);                       // 16 bytes = 32 values
lo    = vec_and(bytes, 0x0F);                 // v0 v2 v4 v6 ...
hi    = vec_sr (bytes, 4);                    // v1 v3 v5 v7 ...
vals  = vec_mergeh(lo, hi);                   // v0 v1 v2 v3 ... in order
```

Widening `uchar → ushort → uint` is done by merging with zeros (little endian),
then `vec_ctf` converts to float. The block scale is constant, so it is folded
in once per block with a single FMA instead of being applied per value.

## Validation

Every change was re-validated before being committed:

- **23/23** against an independent NumPy reference — all block sizes (16/32/64/128),
  with and without explicit zero points, and tile-edge shapes (N=17, N=129,
  K not a multiple of the block size, N=1).
- **17/17** numerical battery — signed int8 (plain `char` is unsigned on PowerPC),
  float16 extremes, endianness, POWER9 VSX quantize kernels.
- **89/89** onnxruntime-genai C++ suite, 0 failures (the 22 skipped need TensorRT
  or ASR model data).
- Real generation: coherent output, deterministic under greedy decoding.

Getting the nibble order or the tile layout wrong does not crash — it returns
plausible, wrong numbers. That is why the NumPy reference was written before any
optimization.

## Repository layout

```
kernel/       the kernel source, as it lands in mlas/lib/power/
patches/      the five commits, as git-format-patch files
build/        build scripts (build, ORT_HOME assembly, progress)
tests/        correctness: NumPy reference + numerical battery
benchmarks/   GEMM microbenchmark and end-to-end thread sweep
docs/         detailed results
```

## Building

Toolchain notes matter here. RHEL 8's system GCC 8.5 ICEs on vectorized code,
and conda's clang 17 fails against conda's libstdc++ 16 headers
(`__builtin_popcountg` is a GCC 14+ / clang 19+ builtin). Use a self-consistent
GCC 13, which also stays within nvcc 12.4's supported host compiler range for a
future CUDA stage.

```bash
conda create -n onnx_build -c conda-forge python=3.12 "cmake=3.31.*" ninja \
        numpy zlib gcc_linux-ppc64le=13.3 gxx_linux-ppc64le=13.3

git clone --recursive -b v1.26.0 https://github.com/microsoft/onnxruntime.git
cd onnxruntime && git am ../patches/*.patch

bash build/build_ort_cpu_power9.sh
```

CMake 4.x removes compatibility with several of ONNX Runtime's pinned
dependencies; 3.31 satisfies the 3.28 minimum without breaking them.

The resulting wheel requires a conda-forge `libstdcxx-ng` at runtime — RHEL 8's
system libstdc++ (GCC 8 era) is too old.

## Status and limitations

- **`SQNBIT_CompInt8` is not implemented.** That is the path AVX2 and NEON use
  for maximum performance: quantize the activations to int8 and use integer dot
  products. POWER9's `vec_msum` would be the natural basis. Note that ONNX
  Runtime only selects it when the `MatMulNBits` node carries
  `accuracy_level=4`; the stock Phi-3-mini export does not.
- **The prefill kernel is still scalar.** Measured not to be the bottleneck —
  int4 already beats fp32 at M=64 — but it is recorded technical debt.
- **POWER10 MMA is unused.** The development machine is POWER9.

Two ideas from the NEON kernel were deliberately not adopted and remain on the
table: converting int→float by placing the nibble directly into a float
mantissa template (no conversion instruction at all), and folding the zero point
into that conversion offset so the subtraction disappears.

## Upstream

The kernel is a candidate for contribution to ONNX Runtime: MLAS simply had no
int4 kernel for POWER, and now it does. It benefits any quantized inference on
Power, not only LLMs.

---

UFCG / IBM — `#ibm-multiarq`
Built and validated on an IBM Power System AC922 (8335-GTH), 2× POWER9,
RHEL 8, ppc64le.

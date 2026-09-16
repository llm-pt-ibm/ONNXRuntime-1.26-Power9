# ONNX Runtime 1.26.0 for POWER9 (ppc64le)

Two build tracks, both producing installable wheels for Python 3.12:

| | Package | Phi-3-mini int4 | What it took |
|---|---|---|---|
| **CPU** | `onnxruntime` | 10.5 tok/s | an int4 MLAS kernel that upstream does not have |
| **GPU** | `onnxruntime_gpu` | **177.8** tok/s (193.5 with cuda graph) | three patches, none ppc64le-specific |

They are **separate packages** — install one or the other, not both.
GPU details in [`gpu/`](gpu/).

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

## GPU (CUDA)

A CUDA build for Tesla V100 (sm_70) also exists, reaching **177.8 tok/s** on the
same model — 17× the best CPU figure, and **193.5** with `enable_cuda_graph=1`.

It needed three patches, none of them ppc64le-specific: an abseil × nvcc 12.4
incompatibility, flash attention defaulting on despite requiring sm_80, and an
undefined `Einsum::DeviceCompute` symbol that makes the CUDA provider fail to
load — **without breaking the build**, so the runtime silently executes on the
CPU while still advertising `CUDAExecutionProvider`.

| | tok/s |
|---|---|
| CPU with the int4 kernel, 80 threads | 10.5 |
| GPU decode | 177.8 |
| GPU decode, `enable_cuda_graph=1` | **193.5** |
| GPU prefill | ~208 |

### What runs where

On GPU the work splits between this project and onnxruntime-genai, and knowing
which is which saves time when something is slow or broken:

| | Where it runs | Fixed for POWER9 by |
|---|---|---|
| Transformer layers, GEMMs, attention | **ONNX Runtime CUDA EP** | the three patches here |
| Sampling, top-k, beam search, KV bookkeeping | genai's own CUDA kernels | nothing — compiled clean |
| Tokenization, chat template | genai, host side | nothing |

genai is not a thin wrapper: it ships its own `.nv_fatbin` with the generation
loop, so that sampling does not force a host round-trip on every token. But the
model itself — everything that dominates the time — is this repository's CUDA
execution provider.

Full write-up, traps and measurements: [`gpu/README.md`](gpu/README.md) and
[`docs/results-gpu.md`](docs/results-gpu.md).

## Repository layout

```
kernel/          the int4 kernel source, as it lands in mlas/lib/power/
patches/         the five CPU commits, as git-format-patch files
cpu/build/       build scripts (build, ORT_HOME assembly, progress)
cpu/tests/       correctness: NumPy reference + numerical battery
cpu/benchmarks/  GEMM microbenchmark and end-to-end thread sweep
gpu/             CUDA build scripts, the three patch scripts, tests
docs/            detailed results, CPU and GPU
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

bash cpu/build/build_ort_cpu_power9.sh
```

CMake 4.x removes compatibility with several of ONNX Runtime's pinned
dependencies; 3.31 satisfies the 3.28 minimum without breaking them.

The resulting wheel requires a conda-forge `libstdcxx-ng` at runtime — RHEL 8's
system libstdc++ (GCC 8 era) is too old.

This produces the `onnxruntime` package. The CUDA build produces
`onnxruntime_gpu` instead — same project, different package name, and they
conflict if both are installed. See [`gpu/README.md`](gpu/README.md).

## Status and limitations

- **`SQNBIT_CompInt8` is not implemented.** That is the path AVX2 and NEON use
  for maximum performance: quantize the activations to int8 and use integer dot
  products. POWER9's `vec_msum` would be the natural basis. Note that ONNX
  Runtime only selects it when the `MatMulNBits` node carries
  `accuracy_level=4`; the stock Phi-3-mini export does not.
- **The prefill kernel is still scalar.** Measured not to be the bottleneck —
  int4 already beats fp32 at M=64 — but it is recorded technical debt.
- **POWER10 MMA is unused.** The development machine is POWER9.
- **On the GPU, the int4 kernel is strictly M=1** — the same gap, mirrored.
  Batch 1 outperforms batch 2 and 4 in total throughput. Writing an `m > 1` int4
  kernel for sm_70 is the highest-return work left.

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

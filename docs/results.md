# Measurements

All numbers were taken on an IBM Power System AC922 (8335-GTH): 2× POWER9,
20 cores per socket, SMT4 (160 logical CPUs), 512 GB RAM, RHEL 8, ppc64le.

Built with conda-forge GCC 13.3, CMake 3.31, Python 3.12.

## Methodology

Two levels of measurement, because they answer different questions.

**`benchmarks/bench_matmul.py`** times a single `MatMulNBits` node against a
plain `MatMul` of the same logical shape, sweeping thread counts. Good for
*diagnosis* — it isolates the kernel.

**`benchmarks/bench_threads.py`** runs real generation through
onnxruntime-genai and reports tokens per second. Good for *decisions* — it is
what a user experiences.

The two disagree, and the second one is the honest one. At optimization step 5
the microbenchmark got slightly worse (67.9 → 62.9 GFLOP/s at N=8192) while
end-to-end generation improved 16% (9.01 → 10.48 tok/s). A single 0.8 ms
operation is dominated by thread-pool overhead and varies substantially between
runs.

For the A/B comparison, two wheels were built from the same source tree with
only the kernel differing, and the md5 of the installed
`onnxruntime_pybind11_state.so` was verified before each measurement. This was
not paranoia: an earlier attempt produced a "without kernel" reading of 23 tok/s
because the wheels had been renamed and `pip` silently refused to install them
(wheel filenames must follow the standard convention), leaving the previous
build in place.

## Optimization progression

Phi-3-mini 3.8B, int4 RTN block-32, decode throughput at the best thread count.

| Step | tok/s | Gain | What changed |
|---|---|---|---|
| Stock, default threads | 0.13 | — | `ComputeBUnpacked` fallback, 160 threads |
| Stock, tuned threads | 0.51 | 3.9× | same fallback, better thread count |
| 1. Scalar kernel | 2.66 | 20× | on the MLAS path; cache-sized tiles, no SIMD |
| 2. VSX unpacking | 7.30 | 56× | 16 nibbles per instruction |
| 3. Four columns | 7.89 | 61× | A was being reloaded once per column |
| 4. Vector accumulator | 9.01 | 69× | no per-block horizontal reduction |
| 5. Full block per load | 10.48 | 81× | `block_size=32` is exactly one 16-byte load |

Step 1 is worth noting: **5× with no SIMD at all**. The gain came purely from
being on the MLAS path, which decompresses in tiles that fit in cache
interleaved with the multiply, instead of materializing the whole matrix.

## Controlled A/B — Llama 3.2 1B Instruct

`onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX`,
`cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4`.

Same machine, same model, same script. Only the presence of the kernel differs.

| Threads | Without kernel | With kernel | Ratio |
|---|---|---|---|
| 20 | 1.50 | 21.61 | 14.4× |
| 40 | 1.45 | 21.97 | 15.2× |
| **80** | 1.28 | **23.49** | **18.4×** |
| 160 (default) | 0.32 | 9.65 | 30.2× |

Best-to-best: **15.7×**.

Note this model requests `accuracy_level=4`, i.e. `SQNBIT_CompInt8`. Since no
CompInt8 kernel exists for POWER, ONNX Runtime falls back to `CompFp32` — the
path this kernel implements. The fallback is silent.

## MatMulNBits in isolation

M=1, K=3072 (Phi-3-mini hidden size), fp32 baseline for reference.

**N=3072**

| Threads | fp32 GFLOP/s | int4 before | int4 after |
|---|---|---|---|
| 1 | 7.8 | 3.7 | 3.1 |
| 16 | 131.4 | 51.8 | 45.0 |
| 20 | 101.0 | 52.8 | 44.9 |
| 80 | 98.7 | 46.1 | 39.3 |

**N=8192**

| Threads | fp32 GFLOP/s | int4 after |
|---|---|---|
| 8 | 45.2 | 29.3 |
| 16 | 86.0 | 56.4 |
| 40 | 109.2 | 61.1 |
| 80 | 140.1 | **62.9–67.9** |

int4 went from 25–45× slower than fp32 (with the fallback) to **1.1–1.8×**.

## Prefill is not a problem

M=64, K=3072, N=8192 — the shape of prompt processing:

| Threads | fp32 GFLOP/s | int4 GFLOP/s | int4/fp32 |
|---|---|---|---|
| 20 | 170.8 | 230.2 | 0.7× |
| 40 | 190.5 | 250.8 | 0.8× |
| 80 | 349.5 | 285.9 | 1.2× |

At M=64 **int4 is faster than fp32** at lower thread counts: B is 8× smaller, so
less memory traffic, and the dequantization cost amortizes over 64 rows. This is
why the prefill kernel was left scalar — it is not the bottleneck.

## Thread count

ONNX Runtime uses all logical CPUs by default. On this machine that is 160, and
it is consistently the worst choice for decoding:

| Threads | Llama 3.2 1B (tok/s) |
|---|---|
| 20 | 21.61 |
| 40 | 21.97 |
| **80** | **23.49** |
| 160 (default) | 9.65 |

The per-token GEMMs of decoding are too small to amortize the synchronization
cost across 160 threads. Set `intra_op_num_threads` between 40 and 80.

Always sweep the thread count before publishing any Power benchmark — the
default will understate the machine by more than 2×.

## Correctness

Re-run after each of the five optimization steps:

| Suite | Result |
|---|---|
| `tests/test_matmulnbits_correctness.py` | 23/23 |
| `tests/test_ort_numerics.py` | 17/17 |
| onnxruntime-genai C++ unit tests | 89 passed, 0 failed, 22 skipped |

The NumPy reference covers block sizes 16/32/64/128, with and without explicit
zero points, and tile-edge shapes: N=17 and N=129 (not multiples of the 16-column
tile), K not a multiple of the block size, N=1, and the real Phi-3-mini decode
shape (M=1, N=3072, K=3072).

Maximum absolute deviation from the reference stayed below 1.3e-5 throughout —
int4 dequantization is exact, so the only error is fp32 accumulation order.

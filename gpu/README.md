# ONNX Runtime 1.26.0 with CUDA on POWER9

CUDA build for Tesla V100 (sm_70), CUDA 12.4.1, cuDNN 9.0.0.312.

**CUDA 12.4.1 is the last release with ppc64le support** — 12.4.0 deprecated
Power, 12.5 removed it. cuDNN 9.0.0.312 is the last ppc64le cuDNN. Nothing newer
is an option on this hardware.

## Results

Phi-3-mini-4k int4 (`cuda/cuda-int4-rtn-block-32`) on a Tesla V100-SXM2-16GB:

| | tok/s |
|---|---|
| CPU with the POWER int4 kernel, 80 threads | 10.5 |
| **GPU, decode** | **177.8** |
| GPU, decode with `enable_cuda_graph=1` | **193.5** |
| GPU, prefill | ~208 |

**17× the best CPU number.** Do not compare against the 0.13 tok/s of stock CPU —
that comparison is true and dishonest at the same time.

See [`../docs/results-gpu.md`](../docs/results-gpu.md) for the full sweep.

## The three patches

None of them is ppc64le-specific. All three are reportable upstream.

### 1. abseil 20250814 × nvcc 12.4 — `patches/patch_abseil_nvcc.sh`

Breaks all 32 `.cu` files of the CUDA EP, with zero errors in the `.cc` files.

In `raw_hash_map.h` the macro uses `IfRRef<int KQual>::AddPtr<K>`. With
`KQual = const &` the alias **collapses to `K`** — a bare template parameter —
and nvcc's EDG frontend inserts a `typename` there. `typename K` is illegal.

Fix in `common.h`:

```cpp
- using AddPtr = Other;
+ using AddPtr = typename std::enable_if<true, Other>::type;
```

A three-line reproducer gives 22 errors under nvcc and 0 under g++.

### 2. Flash attention is ON by default and requires sm_80

ONNX Runtime 1.26 does **not** disable it by architecture — it comes on together
with `USE_CUDA`. On sm_70 it has to be turned off:

```
onnxruntime_USE_FLASH_ATTENTION=OFF
onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=ON   # keep this ON
```

Keeping memory-efficient attention matters: `fmha_sm70.cu` exists and its
dispatch is `sm >= 70`, and it is what backs `GroupQueryAttention` on the V100.
`LEAN_ATTENTION` and `FPA_INTB_GEMM` are already off by default.

### 3. `Einsum::DeviceCompute` undefined in the provider — `patches/patch_ort_einsum_symbol.sh`

The dangerous one: **it does not break the build.**

`provider_bridge_provider.cc` is compiled into the provider `.so` and defines
`Einsum::Compute`. By the key-function rule that emits the vtable of
`onnxruntime::Einsum` there, which needs the address of every virtual — including
`DeviceCompute`, defined only on the CPU side. It is dead code (the CUDA Einsum
is `onnxruntime::cuda::Einsum`, derived from `CudaKernel`) and should vanish
under `--gc-sections`. It does not: on ppc64le the ELFv2 TOC keeps the vtable
alive where x86 discards it.

The fix defines the symbol inside the provider itself as `ORT_NOT_IMPLEMENTED`,
so it fails loudly if the premise ever changes.

Two fixes that do **not** work, both tried first:

- Exporting via `core/providers/cuda/symbols.txt` — the same file feeds
  `generated_source.c`, a **C** file mapping names to addresses. A mangled C++
  name is not a valid C identifier.
- Exporting from `onnxruntime_pybind11_state.so` as well — it compiles and
  genuinely exports (`nm -D` confirms), and still fails, because Python loads
  extensions with `RTLD_LOCAL` so those symbols never reach global scope.

## Traps

### The CUDA EP falls back to CPU silently

The worst failure mode in this port, because nothing breaks. The build passes,
the import passes, `ort.get_available_providers()` lists `CUDAExecutionProvider`
— and the session runs everything on the CPU. A benchmark run this way publishes
a CPU number labelled as GPU.

**Never trust `get_available_providers()`.** After creating a session:

```python
assert sess.get_providers()[0] == "CUDAExecutionProvider"
```

and verify node assignment through `enable_profiling`, counting `args.provider`
per event. `tests/test_ort_cuda.py` does both — that assert caught a false
positive twice during development.

When it happens, the log line that matters is the first one, and it is easy to
lose in a `tail`:

```
[E] TryGetProviderInfo_CUDA ... Failed to load library
    libonnxruntime_providers_cuda.so ... undefined symbol:
    _ZNK11onnxruntime6Einsum13DeviceCompute...
[W] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*.
```

The second line is misleading — it sends you after cuDNN and CUDA, which are
fine.

### Only GPUs 0 and 1 initialize

The box has four V100s, but on this container only two initialize; a bad GPU
takes down `cuInit` for the whole process. Always:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

### The int4 kernel is strictly M=1

`contrib_ops/cuda/quantization/matmul_4bits.cu:340`:

```c
if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) return false;
```

When it refuses, `MatMulNBits` allocates B in fp16 and dequantizes **all** the
weights before a normal GEMM — every token, every one of the 32 layers. So:

| batch | total tok/s | per sequence |
|---|---|---|
| 1 | 171.1 | 171.1 |
| 2 | 70.6 | 35.3 |
| 4 | 139.2 | 34.8 |
| 8 | 268.7 | 33.6 |
| 32 | 917.7 | 28.7 |

**Batch 1 beats batch 2 and 4 in total throughput.** Never batch between 2 and 7.

This is the same problem as the CPU int4 gap, mirrored on the GPU — except here
the fast path exists, it just closes at M=1. Writing an `m > 1` int4 kernel for
sm_70 is the highest-return work left in this port.

### `enable_cuda_graph` ships off

The `genai_config.json` distributed with CUDA models carries
`{"cuda": {"enable_cuda_graph": "0"}}`. Turning it on is **+8.8%** on decode
(177.8 → 193.5) with byte-identical output. Always state whether it was on when
publishing a GPU number.

## Building

```bash
# CUDA 12.4.1 toolkit (last with ppc64le support), keeping the existing driver
sh cuda_12.4.1_*_linux_ppc64le.run --toolkit --silent \
        --toolkitpath=/usr/local/cuda-12.4 --override
ln -sfn lib64 /usr/local/cuda-12.4/lib

bash build_ort_cuda_power9.sh      # four phases, applies both patch scripts
bash make_ort_home_cuda.sh
```

Toolchain is the same as the CPU build — conda GCC 13.3. That is not a
coincidence: nvcc 12.4 rejects `__GNUC__ > 13`, so 13.3 is the last version that
passes. `CMAKE_CUDA_ARCHITECTURES=70`.

## Testing

```bash
export CUDA_VISIBLE_DEVICES=0,1
python tests/test_ort_cuda.py       # provider verification + numerics
python tests/bench_gpu_sweep.py     # batch, prefill, KV growth, cuda graph
```

# GPU measurements

Tesla V100-SXM2-16GB (sm_70), CUDA 12.4.1, cuDNN 9.0.0.312, on an IBM Power
System AC922 (8335-GTH). `CUDA_VISIBLE_DEVICES=0,1`.

Model: Phi-3-mini-4k-instruct, `cuda/cuda-int4-rtn-block-32`.

Two measurement rounds: the original port (2026-09-09) and a revalidation
(2026-09-16). Decode reproduced within noise across both — 178.0 / 178.6 / 178.7
then 177.8 — so the numbers below are stable, not one lucky run.

## Headline

| | tok/s |
|---|---|
| CPU with the POWER int4 kernel, 80 threads | 10.5 |
| GPU decode, stock config | 177.8 |
| GPU decode, `enable_cuda_graph=1` | **193.5** |
| GPU prefill | ~208 |

17× the best CPU figure. Model load: 1.5 s.

## A. Batch

Prompt 128 tokens, 64 generated.

| batch | total tok/s | per sequence | prefill tok/s | spread |
|---|---|---|---|---|
| 1 | 171.1 | 171.1 | 3006.8 | 0.4% |
| 2 | 70.6 | 35.3 | 5212.6 | 0.1% |
| 4 | 139.2 | 34.8 | 5948.3 | 0.2% |
| 8 | 268.7 | 33.6 | 5939.9 | 0.1% |
| 16 | 509.1 | 31.8 | 6025.4 | 0.4% |
| 32 | 917.7 | 28.7 | 6111.3 | 1.4% |

**Batch 1 beats batch 2 and batch 4 in total throughput** — the opposite of what
a GPU should do. Cause: the fast int4 kernel is strictly M=1
(`matmul_4bits.cu:340`). When it refuses, all weights are dequantized to fp16
before a normal GEMM, every token, every layer. That cost is batch-independent,
so per-sequence settles around 35 and the aggregate grows linearly.

Two hypotheses ruled out by measurement rather than reading: XQA
(`ORT_ENABLE_XQA=0` changes nothing, and its default only engages with a
quantized KV cache) and `USE_FPA_INTB_GEMM=OFF` (its `is_supported` requires
`arch >= 75`, so it does not exist on sm_70).

**Operational advice: never use a batch between 2 and 7.**

## B. Prefill vs prompt length

Batch 1.

| prompt | prefill (ms) | tok/s |
|---|---|---|
| 64 | 41 | 1558.1 |
| 256 | 53 | 4847.9 |
| 512 | 91 | 5625.1 |
| 1024 | 177 | 5784.8 |
| 2048 | 371 | 5524.1 |

Saturates around 5.6–5.8k tok/s from prompt 512 onward. Short prompts are
dominated by fixed overhead — at 64 tokens the GPU delivers less than a third of
its plateau rate.

## C. Decode vs KV cache growth

Batch 1, 512 tokens generated.

| position | decode tok/s |
|---|---|
| 64–128 | 177.1 |
| 128–192 | 170.9 |
| 192–256 | 168.5 |
| 256–320 | 163.0 |
| 320–384 | 161.5 |
| 384–448 | 156.0 |
| 448–512 | 154.7 |
| 512–576 | 149.8 |

**15.4% drop** from the first block to the last. KV cache costs 0.375 MB per
token per sequence. Relevant when quoting a number for long-context use: a
figure measured over the first 64 tokens overstates a 2k-token conversation.

## D. CUDA graph

Batch 1, prompt 128, 64 tokens.

| `enable_cuda_graph` | decode tok/s |
|---|---|
| 0 (the shipped default) | 169.8 |
| 1 | 180.5 |

Confirmed in the full end-to-end test as well: 177.8 → 193.5 (**+8.8%**), with
**byte-identical generated text**. The output was compared, not just the
throughput — CUDA graph with dynamic shapes is exactly the kind of thing that
returns wrong results without crashing.

The default comes from the model exporter, not from ONNX Runtime. Always state
whether it was enabled when publishing a GPU number, otherwise comparisons
across machines carry a hidden 9%.

## Verification discipline

Every GPU measurement here was taken through a session that asserts

```python
sess.get_providers()[0] == "CUDAExecutionProvider"
```

and counts per-node provider assignment via `enable_profiling`. A build with the
Einsum symbol bug lists `CUDAExecutionProvider` as available and runs entirely on
the CPU — that assert caught the false positive twice during development.

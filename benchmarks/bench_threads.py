#!/usr/bin/env python
"""
Thread-count sweep for onnxruntime-genai on POWER9.

The box reports 160 CPUs (SMT4 over 40 cores). ORT defaults to using all of
them, which on the small per-token GEMMs of decoding can cost more in
synchronisation than it buys in parallelism. This measures tok/s against
intra_op_num_threads to find where the curve actually peaks.

Usage: python bench_threads.py [model_dir]
"""

import json
import shutil
import sys
import time
from pathlib import Path

import onnxruntime_genai as og

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/root/onnx/models/phi3-mini/cpu_and_mobile/cpu-int4-rtn-block-32")
CONFIG = MODEL_DIR / "genai_config.json"
BACKUP = MODEL_DIR / "genai_config.json.orig"

THREAD_COUNTS = [20, 40, 80, 160]
PROMPT = "<|user|>\nCount from one to twenty in English.<|end|>\n<|assistant|>\n"
N_TOKENS = 20


def set_threads(n):
    cfg = json.loads(BACKUP.read_text())
    so = cfg["model"]["decoder"]["session_options"]
    if n is None:
        so.pop("intra_op_num_threads", None)
    else:
        so["intra_op_num_threads"] = n
    CONFIG.write_text(json.dumps(cfg, indent=4))


def measure():
    t0 = time.time()
    model = og.Model(og.Config(str(MODEL_DIR)))
    tokenizer = og.Tokenizer(model)
    load = time.time() - t0

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=N_TOKENS + 128, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(tokenizer.encode(PROMPT))

    # The first token includes prompt processing (prefill); time it separately.
    t0 = time.time()
    generator.generate_next_token()
    prefill = time.time() - t0

    t0 = time.time()
    n = 0
    while not generator.is_done() and n < N_TOKENS:
        generator.generate_next_token()
        n += 1
    decode = time.time() - t0
    return load, prefill, n / decode if decode else 0.0


def main():
    if not BACKUP.exists():
        shutil.copy(CONFIG, BACKUP)

    print(f"modelo: {MODEL_DIR.name}")
    print(f"{'threads':>8}  {'load(s)':>8}  {'prefill(s)':>11}  {'decode tok/s':>13}")
    print("-" * 48)

    results = []
    try:
        for n in THREAD_COUNTS:
            set_threads(n)
            load, prefill, tps = measure()
            results.append((n, tps))
            print(f"{n:>8}  {load:>8.1f}  {prefill:>11.2f}  {tps:>13.2f}", flush=True)
    finally:
        shutil.copy(BACKUP, CONFIG)

    print("-" * 48)
    best = max(results, key=lambda r: r[1])
    worst = min(results, key=lambda r: r[1])
    print(f"melhor: {best[0]} threads -> {best[1]:.2f} tok/s")
    print(f"pior:   {worst[0]} threads -> {worst[1]:.2f} tok/s")
    if worst[1]:
        print(f"ganho do melhor sobre o pior: {best[1] / worst[1]:.1f}x")


if __name__ == "__main__":
    main()

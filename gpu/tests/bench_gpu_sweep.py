#!/usr/bin/env python
"""
Varredura de desempenho do onnxruntime-genai com CUDA na Tesla V100 (Power9).

Equivalente GPU do bench_threads.py. As dimensoes mudam: em CPU o que dominava era
o numero de threads; em GPU o que domina e batch (ocupacao dos SMs) e o tamanho do
KV cache (banda de memoria). Thread count nao entra -- o grafo inteiro esta na GPU.

Quatro varreduras:
  A. batch          -- decode agregado vs batch. E o ganho que a GPU tem sobre a CPU.
  B. prefill        -- tok/s de prefill vs tamanho do prompt (regime compute-bound).
  C. crescimento KV -- decode tok/s conforme o KV cache cresce (regime memory-bound).
  D. cuda graph     -- on/off, que o genai_config expoe e vem desligado.

Cada configuracao roda um warmup descartado antes de medir: a primeira execucao
carrega o contexto CUDA e faz autotuning de kernel, e contaminaria o primeiro ponto
de cada curva.

Erros de VRAM sao capturados por configuracao (a V100 tem 16 GB e o KV cache do
Phi-3 custa ~0,375 MB/token/sequencia) -- uma config que nao cabe vira uma linha
"OOM" na tabela em vez de matar a varredura inteira.

Uso: python bench_gpu_sweep.py [model_dir] [--json saida.json]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import onnxruntime_genai as og

DEFAULT_MODEL = "/root/onnx/models/phi3-mini/cuda/cuda-int4-rtn-block-32"

# Texto de enchimento para montar prompts de tamanho controlado em tokens.
FILLER = (
    "A arquitetura POWER9 da IBM foi projetada para cargas de trabalho de alto "
    "desempenho, com multiplos nucleos, SMT4 e barramento NVLink para aceleradores. "
)


def kv_mb_per_token(cfg):
    """KV cache por token por sequencia, em MB (fp16, k e v)."""
    d = cfg["model"]["decoder"]
    return d["num_key_value_heads"] * d["head_size"] * d["num_hidden_layers"] * 2 * 2 / 2**20


class Runner:
    """Mantem um Model carregado entre as medicoes (recarregar custa ~1,4 s)."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = og.Model(og.Config(str(self.model_dir)))
        self.tokenizer = og.Tokenizer(self.model)

    def tokens_of_length(self, n):
        """Lista de exatamente n tokens, montada por enchimento e corte."""
        text = FILLER * (n // 10 + 2)
        toks = self.tokenizer.encode(f"<|user|>\n{text}<|end|>\n<|assistant|>\n")
        toks = list(toks)
        if len(toks) < n:
            toks = (toks * (n // len(toks) + 1))[:n]
        return toks[:n]

    def run(self, batch, prompt_len, n_new, window=None):
        """Uma medicao. Devolve dict com prefill_s e tok/s de decode.

        Se window for dado, tambem devolve o tok/s por janela de `window` tokens,
        para expor o efeito do crescimento do KV cache.
        """
        toks = self.tokens_of_length(prompt_len)
        batch_tokens = [toks] * batch

        params = og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=prompt_len + n_new, do_sample=False, batch_size=batch
        )
        generator = og.Generator(self.model, params)

        t0 = time.perf_counter()
        generator.append_tokens(batch_tokens)
        prefill_s = time.perf_counter() - t0

        windows = []
        t_start = time.perf_counter()
        t_win = t_start
        n = 0
        while not generator.is_done() and n < n_new:
            generator.generate_next_token()
            n += 1
            if window and n % window == 0:
                now = time.perf_counter()
                windows.append(window * batch / (now - t_win))
                t_win = now
        decode_s = time.perf_counter() - t_start

        return {
            "batch": batch,
            "prompt_len": prompt_len,
            "n_new": n,
            "prefill_s": prefill_s,
            "prefill_tps": batch * prompt_len / prefill_s if prefill_s else 0.0,
            "decode_tps_total": n * batch / decode_s if decode_s else 0.0,
            "decode_tps_per_seq": n / decode_s if decode_s else 0.0,
            "windows": windows,
        }

    def measure(self, batch, prompt_len, n_new, repeats=3, window=None):
        """Warmup descartado + `repeats` medicoes; devolve a mediana."""
        self.run(batch, min(prompt_len, 32), 4)  # warmup barato
        runs = [self.run(batch, prompt_len, n_new, window) for _ in range(repeats)]
        best = dict(runs[0])
        for key in ("prefill_tps", "decode_tps_total", "decode_tps_per_seq", "prefill_s"):
            best[key] = statistics.median(r[key] for r in runs)
        best["spread_pct"] = (
            100
            * (max(r["decode_tps_total"] for r in runs) - min(r["decode_tps_total"] for r in runs))
            / best["decode_tps_total"]
            if best["decode_tps_total"]
            else 0.0
        )
        return best


def guarded(fn):
    """Roda uma medicao devolvendo o erro em vez de propagar (tipicamente VRAM)."""
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - queremos qualquer falha como dado
        msg = str(exc).strip().splitlines()[0][:80]
        return None, msg


def sweep_batch(runner, results):
    print("\n=== A. batch (prompt 128 tok, 64 tokens gerados) ===")
    print(f"{'batch':>6}  {'decode tok/s':>13}  {'por seq':>9}  {'prefill tok/s':>14}  {'spread':>7}")
    print("-" * 60)
    for batch in [1, 2, 4, 8, 16, 32]:
        r, err = guarded(lambda b=batch: runner.measure(b, 128, 64))
        if err:
            print(f"{batch:>6}  {'FALHOU':>13}  {err}")
            results.setdefault("batch", []).append({"batch": batch, "error": err})
            break
        results.setdefault("batch", []).append(r)
        print(
            f"{batch:>6}  {r['decode_tps_total']:>13.1f}  {r['decode_tps_per_seq']:>9.1f}"
            f"  {r['prefill_tps']:>14.1f}  {r['spread_pct']:>6.1f}%",
            flush=True,
        )


def sweep_prefill(runner, results):
    print("\n=== B. prefill vs tamanho do prompt (batch 1) ===")
    print(f"{'prompt':>7}  {'prefill(ms)':>12}  {'prefill tok/s':>14}")
    print("-" * 40)
    for plen in [64, 256, 512, 1024, 2048]:
        r, err = guarded(lambda p=plen: runner.measure(1, p, 8))
        if err:
            print(f"{plen:>7}  {'FALHOU':>12}  {err}")
            results.setdefault("prefill", []).append({"prompt_len": plen, "error": err})
            break
        results.setdefault("prefill", []).append(r)
        print(f"{plen:>7}  {r['prefill_s'] * 1000:>12.0f}  {r['prefill_tps']:>14.1f}", flush=True)


def sweep_kv_growth(runner, results):
    print("\n=== C. decode vs crescimento do KV cache (batch 1, 512 tokens) ===")
    r, err = guarded(lambda: runner.measure(1, 64, 512, repeats=1, window=64))
    if err:
        print(f"FALHOU: {err}")
        results["kv_growth"] = {"error": err}
        return
    results["kv_growth"] = r
    print(f"{'posicao':>18}  {'decode tok/s':>13}")
    print("-" * 35)
    for i, tps in enumerate(r["windows"]):
        lo, hi = 64 + i * 64, 64 + (i + 1) * 64
        print(f"{f'{lo}-{hi}':>18}  {tps:>13.1f}")
    if len(r["windows"]) >= 2:
        queda = 100 * (1 - r["windows"][-1] / r["windows"][0])
        print(f"queda do primeiro ao ultimo bloco: {queda:.1f}%")


def sweep_cuda_graph(runner_factory, model_dir, results):
    print("\n=== D. cuda graph (batch 1, prompt 128, 64 tokens) ===")
    cfg_path = Path(model_dir) / "genai_config.json"
    backup = cfg_path.with_suffix(".json.bench_backup")
    if not backup.exists():
        backup.write_text(cfg_path.read_text())
    print(f"{'cuda_graph':>11}  {'decode tok/s':>13}")
    print("-" * 28)
    try:
        for flag in ["0", "1"]:
            cfg = json.loads(backup.read_text())
            opts = cfg["model"]["decoder"]["session_options"]["provider_options"]
            for entry in opts:
                if "cuda" in entry:
                    entry["cuda"]["enable_cuda_graph"] = flag
            cfg_path.write_text(json.dumps(cfg, indent=4))

            r, err = guarded(lambda: runner_factory().measure(1, 128, 64))
            if err:
                print(f"{flag:>11}  {'FALHOU':>13}  {err}")
                results.setdefault("cuda_graph", []).append({"flag": flag, "error": err})
                continue
            r["cuda_graph"] = flag
            results.setdefault("cuda_graph", []).append(r)
            print(f"{flag:>11}  {r['decode_tps_total']:>13.1f}", flush=True)
    finally:
        cfg_path.write_text(backup.read_text())
        backup.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = json.loads((Path(args.model_dir) / "genai_config.json").read_text())
    print(f"modelo: {Path(args.model_dir).name}")
    print(f"onnxruntime_genai: {og.__version__}")
    print(f"KV cache: {kv_mb_per_token(cfg):.3f} MB por token por sequencia")

    results = {
        "model": str(args.model_dir),
        "genai_version": og.__version__,
        "kv_mb_per_token": kv_mb_per_token(cfg),
    }

    runner = Runner(args.model_dir)
    sweep_batch(runner, results)
    sweep_prefill(runner, results)
    sweep_kv_growth(runner, results)
    del runner  # D recarrega o modelo a cada flag, pois a opcao e de sessao
    sweep_cuda_graph(lambda: Runner(args.model_dir), args.model_dir, results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nresultados em {args.json}")


if __name__ == "__main__":
    sys.exit(main())

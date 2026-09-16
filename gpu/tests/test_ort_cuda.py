"""Valida o CUDA EP do ORT construido para ppc64le/sm_70.

Nao basta o provider aparecer na lista: um build quebrado lista o CUDAExecutionProvider
e silenciosamente executa tudo na CPU. Por isso aqui a gente (1) forca o CUDA EP,
(2) confere a atribuicao de no por no no profile, e (3) compara o resultado com numpy.
"""
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

print("onnxruntime:", ort.__version__)
print("providers disponiveis:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP ausente"

# Modelo minimo: C = A @ B + bias, em float16 (o dtype que o genai vai usar).
N = 512
A = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [N, N])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT16, [N, N])
C = helper.make_tensor_value_info("C", TensorProto.FLOAT16, [N, N])
bias = helper.make_tensor(
    "bias", TensorProto.FLOAT16, [N], np.ones(N, dtype=np.float16).tobytes(), raw=True
)
graph = helper.make_graph(
    [
        helper.make_node("MatMul", ["A", "B"], ["mm"]),
        helper.make_node("Add", ["mm", "bias"], ["C"]),
    ],
    "matmul_add",
    [A, B],
    [C],
    [bias],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

so = ort.SessionOptions()
so.enable_profiling = True
so.log_severity_level = 2
sess = ort.InferenceSession(
    model.SerializeToString(), so, providers=["CUDAExecutionProvider"]
)
print("providers da sessao:", sess.get_providers())
assert sess.get_providers()[0] == "CUDAExecutionProvider", "sessao nao usou CUDA"

rng = np.random.default_rng(0)
a = rng.standard_normal((N, N), dtype=np.float32).astype(np.float16)
b = rng.standard_normal((N, N), dtype=np.float32).astype(np.float16)
out = sess.run(["C"], {"A": a, "B": b})[0]

ref = (a.astype(np.float32) @ b.astype(np.float32)) + 1.0
err = np.abs(out.astype(np.float32) - ref).max() / np.abs(ref).max()
print(f"erro relativo maximo vs numpy: {err:.2e}")

prof = sess.end_profiling()
import json

with open(prof) as f:
    events = json.load(f)
eps = {}
for e in events:
    p = (e.get("args") or {}).get("provider")
    if p:
        eps[p] = eps.get(p, 0) + 1
print("nos executados por provider:", eps)

assert err < 1e-2, f"resultado numericamente errado: {err}"
assert any("CUDA" in k for k in eps), f"nenhum no rodou no CUDA EP: {eps}"
print("\nOK: CUDA EP carregou, executou na GPU e bateu com numpy.")

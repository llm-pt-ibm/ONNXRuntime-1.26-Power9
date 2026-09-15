# Provenance and licensing

The kernel in `kernel/` and the patches in `patches/` are derived from
**ONNX Runtime**, which is licensed under the MIT License
(Copyright (c) Microsoft Corporation).

They are modifications to files in `onnxruntime/core/mlas/` and are intended to
be contributed back upstream. They carry the same MIT License and retain the
original copyright headers.

This repository adds build scripts, tests, benchmarks and documentation
produced by the UFCG / IBM multi-architecture team. Those are released under
the same terms so the whole thing can be upstreamed without friction.

Nothing here vendors or redistributes ONNX Runtime itself — the build scripts
fetch it from the official repository at tag `v1.26.0`.

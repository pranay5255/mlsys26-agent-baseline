# Installation And Runtime

## Supported Environment

NVIDIA documents CUTLASS DSL 4.4 as supporting Linux and Python 3.10 through
3.14. The CuTe DSL wheel is matched to CUDA Toolkit families. For this repo's
Modal runtime, use CUDA 13.1 and the CUDA 13 extra:

```bash
uv sync
uv run python -c "import cutlass, cutlass.cute; print('cutlass ok')"
```

The project dependency is `nvidia-cutlass-dsl[cu13]`. Local development should
use `uv sync` so `uv.lock` remains the source of reproducible dependency state.

## Modal B200

The Modal image should start from a CUDA 13.1 devel image. Use the same Python
minor version as the local Modal client when `serialized=True` is used; this
repo currently uses Python 3.11 locally, so the Modal image should also use
Python 3.11. Install:

- `flashinfer-bench`
- `torch`
- `pydantic`
- `nvidia-cutlass-dsl[cu13]`

The runtime should set:

- `CUTE_DSL_CACHE_DIR=/tmp/cute_dsl_cache`
- `CUTE_DSL_LINEINFO=1` only in debug mode
- `CUTE_DSL_LOG_TO_CONSOLE=1` and `CUTE_DSL_LOG_LEVEL=20` only in debug mode

## Driver And Cache Notes

CuTe DSL generates CUDA kernels through MLIR and ptxas. The wheel requires a
compatible NVIDIA driver for its CUDA Toolkit family. JIT compilation creates
executors and may serialize cache entries under `/tmp` unless
`CUTE_DSL_CACHE_DIR` is set to a persistent writable path.

## Quick Checks

```bash
uv run python - <<'PY'
import importlib.metadata as md
import cutlass
import cutlass.cute as cute
from flashinfer_bench.data import SupportedLanguages

print(md.version("nvidia-cutlass-dsl"))
print(SupportedLanguages.PYTHON)
PY
```

# Programming Model

## Core Idea

CuTe DSL is a Python embedded DSL for authoring GPU kernels using CUTLASS/CuTe
concepts. Python code is transformed into IR and JIT-compiled into CUDA device
code. The main abstractions are layouts, tensors, atoms, tiled operations, and
explicit hardware/thread hierarchy control.

## Decorators

- `@cute.jit`: host-side JIT function. It can be called from Python and can
  launch kernels. It is the right place for metaprogramming, layout setup, and
  calling `@cute.kernel`.
- `@cute.kernel`: GPU kernel function. It compiles into a specialized GPU
  symbol and must be launched from a JIT host function or compatible compiled
  executor.

Decorated functions must live in normal Python source files so CuTe DSL can
inspect their source. REPL-style `python -c`, `exec`, or dynamically generated
decorated functions can fail source inspection. Generated agent solutions are
safe because FlashInfer Bench imports them from `main.py`.

Typical generated solution shape:

```python
import torch
import cutlass
import cutlass.cute as cute


@cute.kernel
def device_kernel(...):
    ...


@cute.jit
def launch_kernel(..., stream):
    device_kernel(...).launch(
        grid=[...],
        block=[...],
        stream=stream,
    )


def run(*args, **kwargs):
    ...
```

Do not rely on JIT/kernel return values for benchmark outputs. Kernels should
write into output tensors, and the plain Python `run` wrapper should return the
reference-compatible result object.

## Calling Rules

- Python can call `@cute.jit`.
- A plain Python function cannot directly call `@cute.kernel`.
- `@cute.jit` can call `@cute.jit` and `@cute.kernel`.
- `@cute.kernel` can call helper Python functions or `@cute.jit` only when they
  resolve at compile time in supported patterns.
- `@cute.kernel` cannot call another `@cute.kernel`.

## Launch Parameters

Kernel launch accepts:

- `grid`: CTA grid dimensions.
- `block`: block dimensions.
- `cluster`: preferred cluster dimensions when used.
- `fallback_cluster`: minimum guaranteed cluster size.
- `smem`: dynamic shared memory bytes. `None` lets CuTe calculate when possible.
- `max_number_threads`, `min_blocks_per_mp`, `use_pdl`, `cooperative`: advanced
  launch hints.

For this agent, prompts should require explicit `grid`, `block`, and `stream`.
Use `cutlass.cuda.default_stream()` or a stream passed by the caller. Only use
cluster/shared-memory options when the algorithm needs them.

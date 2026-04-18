# Control Flow And Caching

## Hybrid Compilation

CuTe DSL combines AST rewriting and tracing:

- AST rewriting preserves Python control flow structure such as loops and
  branches.
- Tracing records tensor/arithmetic operations inside those structured regions.

This means generated code should use simple, explicit GPU control flow and avoid
Python features that depend on full Python runtime semantics inside JIT regions.

## Compile-Time Vs Runtime

- `cutlass.Constexpr` marks a static compile-time argument.
- `cutlass.const_expr(condition)` forces compile-time evaluation.
- A normal `if pred` emits runtime IR control flow.
- `cutlass.range_constexpr` is for compile-time loops.
- `range` and `cutlass.range` are runtime loop forms; `cutlass.range` also
  supports advanced unrolling and pipelining attributes.

## Unsupported Control-Flow Patterns

Avoid these in `@cute.jit` or `@cute.kernel` dynamic regions:

- `break`, `continue`, or `pass`
- raising exceptions
- returning from inside dynamic control flow
- relying on values defined only inside a dynamic branch outside that branch
- changing the type of a variable across dynamic branches

## JIT Executor Cache

`cute.compile(fn, *args, options=...)` compiles and returns a callable JIT
executor. The executor can be cached by generated Python code. Cache keys should
include:

- input shapes
- strides/layout kind
- dtypes
- device
- tuning parameters such as tile sizes, cluster shape, vector width, and epilogue
  options

CuTe DSL also has an internal cache, but generated agent code should keep its own
module-level cache so warmup can pay compilation cost once and timed iterations
can reuse the same executor.

## Cache Environment

- `CUTE_DSL_DISABLE_FILE_CACHING=True`: disable file cache.
- `CUTE_DSL_CACHE_DIR=/path`: choose persistent file cache directory.

In Modal, use a writable `/tmp` path unless a persistent volume cache is
explicitly added.

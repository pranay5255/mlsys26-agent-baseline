# Debugging And Limitations

## Debug Env Vars

- `CUTE_DSL_LINEINFO=1`: generate line information.
- `CUTE_DSL_LOG_TO_CONSOLE=1`: log to console.
- `CUTE_DSL_LOG_TO_FILE=path`: log to a file.
- `CUTE_DSL_LOG_LEVEL=20`: info logging; 10 is debug.
- `CUTE_DSL_PRINT_IR=1`: print generated IR.
- `CUTE_DSL_KEEP_IR=1`: keep generated IR.
- `CUTE_DSL_KEEP_PTX=1`: keep generated PTX.
- `CUTE_DSL_KEEP_CUBIN=1`: keep generated CUBIN.
- `CUTE_DSL_DUMP_DIR=path`: choose dump directory.

Compiled executors expose generated artifacts programmatically through attributes
such as `__ptx__`, `__cubin__`, and `__mlir__`.

## Print Behavior

- Python `print()` in JIT code is compile-time only.
- `cute.printf()` prints at GPU runtime and changes generated code. It should be
  removed or guarded before performance measurements.

## Important Limitations

- CuTe DSL is not full Python inside JIT compilation.
- Python lists, tuples, and dicts are compile-time structures, not mutable
  dynamic runtime containers.
- Dynamic list indexing is not supported in the general Python sense.
- `break`, `continue`, `pass`, exceptions, and returns from dynamic control-flow
  bodies are unsupported.
- Avoid passing dynamic values through mutable object state.
- The special `_` variable is treated as ignored and must not be read.
- CuTe layout algebra is available in JIT-compiled contexts; native Python usage
  is restricted.
- Current layout algebra supports 32-bit shapes/strides.
- CuTe DSL source inspection expects decorated functions to come from a real
  source file. REPL-style dynamic definitions can fail before compilation.
- JIT return-value support is limited; generated benchmark solutions should
  write into tensors and return from the outer Python `run` wrapper.

## Triage Order

1. Python syntax/import error.
2. CuTe DSL type or unsupported-Python error.
3. Layout/static-vs-dynamic mismatch.
4. Kernel launch/grid/block/smem error.
5. Numerical correctness error.
6. Performance regression.

The tuner prompt should feed the exact error log and ask for localized
`str_replace` edits.

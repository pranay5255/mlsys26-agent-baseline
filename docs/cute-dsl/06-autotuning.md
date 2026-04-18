# Autotuning

## CuTe DSL Tuning Pattern

NVIDIA's autotuning guidance is:

1. Define a valid search space.
2. Compile and benchmark each configuration.
3. Cache compiled executors and selected best configs.

For this repo, the benchmark already has a warmup phase. Generated `run` code may
use warmup calls to compile and select among a small bounded set of variants, but
timed iterations must reuse cached compiled executors.

## GEMM-Oriented Parameters

For Blackwell GEMM examples, useful search dimensions include:

- MMA tile shape for M/N/K.
- CTA cluster shape.
- whether to use Blackwell 2-CTA instructions.
- whether to use TMA stores.
- epilogue fusion choices.

For non-GEMM workloads, analogous parameters are:

- block size and vector width.
- elements per thread.
- shared memory staging depth.
- whether to fuse adjacent PyTorch ops.
- static vs dynamic layout choice.

## Cache Keys

Use two cache layers when useful:

- compile cache key: dtype, full layout/shape/stride if static, and all kernel
  tuning parameters.
- input selection key: shape/dtype/layout family used to choose the best compiled
  executor.

Avoid overly broad keys. A compiled executor for a static layout must not be
reused for incompatible shapes or strides.

## Benchmarking

FlashInfer Bench performs correctness, warmup, and timed iterations. The agent
passes benchmark counts into prompts. CuTe generated code should avoid retuning
inside timed iterations unless the cache misses because the shape/layout key is
new.

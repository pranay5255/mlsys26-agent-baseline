# Proposer prompt: generate a new CuTe DSL solution proposal

import re

from agent.utils import CUTE_DSL_SOURCE_SCAFFOLDING

PROBLEM_STATEMENT = """## Problem Statement

You write custom CuTe DSL GPU kernels to replace the PyTorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace.
You may make the decision to replace some operators with custom kernels and leave others unchanged.
You may replace multiple operators with custom implementations,
consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu),
or algorithmic changes (such as online softmax). You are only limited by your imagination.

The goal is to get the best performance for the given architecture.

"""

POOL_PROMPT = """## Solution Pool

Here are community-developed CuTe DSL solutions and their runtime metrics. You can reference them to generate your own solution.

DO NOT directly copy these solutions. Generate your own solution with meaningful modifications,
such as changing the operator, changing the algorithm, fixing the correctness errors, etc.

<Solutions and their Runtime Metrics>
{pool_kernels_and_metrics}
</Solutions and their Runtime Metrics>

Now, please generate your own CuTe DSL solution with the community-developed solutions as reference:
"""


CUTE_DSL_PROMPT = """Generate a complete CuTe DSL Python solution optimized for NVIDIA {target_gpu} GPU ({gpu_architecture}) for

{definition}

CuTe DSL package version available to the agent environment: {cute_dsl_version}

Requirements:
- Write clean, efficient CuTe DSL code optimized for {target_gpu} / {gpu_architecture}.
- Use this exact import pattern when using DLPack:
  - import torch
  - import cutlass
  - import cutlass.cute as cute
  - from cutlass.cute.runtime import from_dlpack
- Do not import from_dlpack from cutlass or cutlass.cute; those imports fail in this environment.
- Do not use "from cutlass.cuda import default_stream" because cutlass.cuda is an attribute, not an importable module. Use cutlass.cuda.default_stream() and cutlass.cuda.stream_sync(stream).
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized. Your CuTe DSL implementation must match its computational accuracy while delivering high performance.
- Use the definition's tensor shapes, dtypes, and axes information to guide layouts, tiling, memory access patterns, and launch parameters.
- Use CuTe DSL decorators correctly:
  - Use @cute.kernel for GPU device kernels.
  - Use @cute.jit for host-side launch/compile wrappers.
  - Launch kernels with explicit grid, block, optional cluster, optional smem, and stream.
  - Do not use Triton/CUDA-style bracket launch syntax like kernel[grid, block](...). Use kernel(args...).launch(grid=..., block=..., stream=...) inside a @cute.jit launcher.
  - Define decorated functions at module scope in the generated file; CuTe DSL cannot parse functions created dynamically in REPL-style exec/eval contexts.
  - Do not depend on return values from JIT/device functions. Write GPU results into output tensors and return from the Python run wrapper.
- Use CuTe DSL tensor/layout APIs deliberately:
  - Convert PyTorch tensors through CuTe runtime/DLPack APIs or direct pointers.
  - Use CuTe DSL device builtins from cute.arch, such as cute.arch.thread_idx(), cute.arch.block_idx(), cute.arch.block_dim(), and cute.arch.sync_threads(); do not use CUDA/Triton-style names like cute.threadIdx, cute.blockIdx, cute.blockDim, cute.syncthreads, cute.load, cute.store, or cute.shared_memory.
  - The evaluator performs a static CuTe preflight and rejects code using cute.load, cute.store, cute.shared_memory, or bracket launch before spending a B200 compile.
  - Choose static layouts when shapes are fixed and dynamic layouts when the same compiled function should handle multiple shapes.
  - Use cutlass.Constexpr for compile-time parameters and normal annotated values for runtime parameters.
  - Avoid unsupported dynamic Python behavior inside JIT/kernel functions, including mutating Python list/dict structure with runtime values, dynamic list indexing, exceptions from dynamic control-flow regions, break/continue in dynamic loops, or reading the special _ variable.
- Cache compiled CuTe executors at module scope using keys derived from input shape, dtype, stride/layout, device, and kernel tuning parameters. cute.compile may be used during warmup, but repeated benchmark iterations must reuse cached compiled executors.
- The benchmark will run correctness first, then {benchmark_warmup_runs} warmup runs, then {benchmark_iterations} timed iterations for {benchmark_trials} trial(s), with a timeout of {eval_timeout_seconds} seconds.
- It is acceptable to use torch only for allocation, fallback shape/device handling, and reference-compatible glue. The performance-critical replaced operations should run through CuTe DSL kernels.

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the CuTe DSL compiled launcher/kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/CuTe DSL syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/CuTe DSL code
- NO Triton imports, decorators, tl language calls, or triton.jit kernels
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

{cute_dsl_scaffold}

Generate complete, runnable code only - no framework will add device handling wrapper code.

Generate the implementation:
"""


def _format_experiment_focus(task_params: dict | None) -> str:
    """Return optional experiment-specific prompt guidance."""
    if not task_params:
        return ""
    experiment_focus = task_params.get("experiment_focus")
    if not experiment_focus:
        return ""
    return (
        "\n## Experiment Focus\n\n"
        "For this run, bias the solution design toward the following objective "
        "while preserving the exact definition semantics:\n\n"
        f"{experiment_focus}\n"
    )


def generate_proposer_prompt(pool_prompt: str = None, task_params: dict = None):
    prompt = PROBLEM_STATEMENT

    # Fill template with task_params
    format_params = {
        **task_params,
        "cute_dsl_scaffold": CUTE_DSL_SOURCE_SCAFFOLDING,
    }
    required_keys = set(re.findall(r"\{(\w+)\}", CUTE_DSL_PROMPT))
    for key in required_keys:
        if key not in format_params:
            raise ValueError(f"Missing required parameter: {key}")

    prompt += CUTE_DSL_PROMPT.format(**format_params)
    prompt += _format_experiment_focus(task_params)
    if pool_prompt is not None:
        prompt += pool_prompt
    return prompt


def generate_pool_prompt_single(
    pool_kernels: list,
    pool_metrics: list,
    *,
    proposal_ids: list[int] | None = None,
):
    if len(pool_kernels) == 0:
        return ""
    pool_kernels_and_metrics = "\n\n".join(
        [
            (
                f"\n### {i}-th kernel"
                + (
                    f" (proposal_id={proposal_ids[i]})"
                    if proposal_ids is not None and i < len(proposal_ids)
                    else ""
                )
                + ":\n\n```python\n"
                + f"{kernel}\n"
                + "```\n\n"
                + f"### {i}-th metrics:\n{metric}"
            )
            for i, (kernel, metric) in enumerate(zip(pool_kernels, pool_metrics))
        ]
    )
    return POOL_PROMPT.format(pool_kernels_and_metrics=pool_kernels_and_metrics)


def generate_pool_prompt(
    *,
    kernel_pool: list,
    metrics_pool: list,
    kernel_pool_ids: list[int] | None = None,
    elite_kernel_pool: list | None = None,
    elite_metrics_pool: list | None = None,
    elite_pool_ids: list[int] | None = None,
):
    """
    Build a single Kernel Pool prompt that can include both a "recent/trajectory" pool
    and an "elite/best" pool. Any pool can be empty/None.
    """
    elite_kernel_pool = elite_kernel_pool or []
    elite_metrics_pool = elite_metrics_pool or []

    parts = []

    elite_part = generate_pool_prompt_single(
        elite_kernel_pool, elite_metrics_pool, proposal_ids=elite_pool_ids
    )
    if elite_part:
        inner = elite_part.split("<Solutions and their Runtime Metrics>")[1].split(
            "</Solutions and their Runtime Metrics>"
        )[0]
        parts.append(("## Context type: elite\n\n" + inner.strip()).strip())

    recent_part = generate_pool_prompt_single(
        kernel_pool, metrics_pool, proposal_ids=kernel_pool_ids
    )
    if recent_part:
        # Strip the outer POOL_PROMPT wrapper so we can merge into one wrapper.
        inner = recent_part.split("<Solutions and their Runtime Metrics>")[1].split(
            "</Solutions and their Runtime Metrics>"
        )[0]
        parts.append(("## Context type: recent\n\n" + inner.strip()).strip())

    if len(parts) == 0:
        return ""

    merged = "\n\n".join(parts)
    return POOL_PROMPT.format(pool_kernels_and_metrics=merged)

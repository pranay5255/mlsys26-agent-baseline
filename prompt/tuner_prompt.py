# Tuner prompt: refine an existing CuTe DSL solution with str_replace edits

import re

from agent.eval import EvalResult
from agent.utils import CUTE_DSL_SOURCE_SCAFFOLDING

_HARDWARE_INFO = """## Hardware Information

Here is some information about the underlying hardware that you should keep in mind:

- The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

"""

PROBLEM_STATEMENT = """## Problem Statement

You tune custom CuTe DSL Python solutions in the given architecture to get better performance. The architecture is the reference architecture, and the custom CuTe DSL solutions are the previous solutions you have generated.

"""

TASK_INSTRUCTION = """## Task Instruction
You are given the following kernel definition:

```python
{definition}
```

The input shapes can be found in the input of the architecture, and the dtype is {dtype_str}.

CuTe DSL package version available to the agent environment: {cute_dsl_version}

The tuning metrics contain the following information:

* **Compiled**: whether the kernel is compiled successfully
* **Error Message**: the compilation or runtime error encountered by the kernel (if any)
* **Correctness**: whether the kernel is correct
* **Runtime**: the runtime of the kernel
* **Fast_p**: compared with the standard PyTorch implementation, how much speedup the customized kernel achieves, calculated as *standard time / custom time*.

### Test Conditions

* **Correctness Test:**
  First, verify the correctness of the custom kernels by running each kernel with the specified input shapes and data types.

* **Warm-up Phase:**
  Warm up the solution by running it {benchmark_warmup_runs} time(s) with the same input shapes and data types.
  The runtime during the warm-up phase is **not** included in the final runtime, so you may include CuTe DSL compilation or bounded auto-tuning code as part of this phase.

* **Performance Test:**
  Finally, test the performance of the custom solution for {benchmark_iterations} timed iteration(s) and {benchmark_trials} trial(s), with a timeout of {eval_timeout_seconds} seconds.
  The runtime from these test runs **is included** in the final performance measurement. Reuse cached CuTe DSL compiled executors in this phase.

### CuTe DSL Requirements

- Keep the solution as complete, runnable Python exposing a "run" entry point.
- Use CuTe DSL APIs correctly: @cute.jit for host launchers, @cute.kernel for GPU kernels, explicit launch grid/block/stream, and module-level cute.compile caches when compiling explicitly.
- Do not use Triton/CUDA-style bracket launch syntax like kernel[grid, block](...). Use kernel(args...).launch(grid=..., block=..., stream=...) inside a @cute.jit launcher.
- Import DLPack conversion only as `from cutlass.cute.runtime import from_dlpack`; never use `from cutlass import from_dlpack` or `from cutlass.cute import from_dlpack`.
- Use stream helpers as `cutlass.cuda.default_stream()` and `cutlass.cuda.stream_sync(stream)`; never use `from cutlass.cuda import default_stream` or `stream.synchronize()`.
- Use device builtins from `cute.arch`, such as `cute.arch.thread_idx()`, `cute.arch.block_idx()`, `cute.arch.block_dim()`, and `cute.arch.sync_threads()`; do not use CUDA/Triton-style names like `cute.threadIdx`, `cute.blockIdx`, `cute.blockDim`, `cute.syncthreads`, `cute.load`, `cute.store`, or `cute.shared_memory`.
- The evaluator performs a static CuTe preflight and rejects code using `cute.load`, `cute.store`, `cute.shared_memory`, or bracket launch before spending a B200 compile.
- Keep decorated CuTe functions at module scope; do not dynamically create them with exec/eval or nested REPL-style code.
- Do not depend on JIT/kernel return values for outputs. Write GPU results into tensors and return from the Python run wrapper.
- Preserve device handling: accept CPU or CUDA tensors, move work tensors to CUDA if needed, call CuTe DSL kernels on CUDA tensors, and return outputs on the original device.
- Do not introduce Triton imports, triton.jit, triton.language as tl, or tl operations.
- Avoid unsupported dynamic Python semantics in JIT/kernel functions, including dynamic mutation of Python containers, dynamic list indexing, break/continue in dynamic loops, exceptions from dynamic control-flow regions, and reading the special _ variable.
- If correcting a compile error, prefer small changes to imports, annotations, static/dynamic layout handling, stream handling, launch parameters, or cache key construction before changing the algorithm.

{cute_dsl_scaffold}

### Goal

- Perform small, localized updates to code in the last version of the custom CuTe DSL solution with the str_replace command to correct correctness errors or improve performance, but keep the high-level architecture unchanged.
When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation.
- You should ensure the `old_str` matches exactly with the file content, otherwise the str_replace tool will fail.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: You should prefer to send all edits in a single message with multiple calls rather than multiple messages with a single call each.

#### **Output Format**:

You should output all the edits in a single message with multiple call. Each call should be a single edit as follows with id `1` to `n`.
- For each edit, you should provide the reasoning for the edit in the <reasoning_i> block, followed by the old code block in the <old_str_i> block, followed by the new code block in the <new_str_i> block.
- You should ensure the `old_str_i` matches exactly with the file content, otherwise the str_replace tool will fail.

Example output format:

<reasoning_1>
// reasoning for the edit 1
...
</reasoning_1>
<old_str_1>
// old code block (must match exactly)
...
</old_str_1>
<new_str_1>
// new code block
...
</new_str_1>

...

<reasoning_n>
// reasoning for the edit n
...
</reasoning_n>
<old_str_n>
// old code block (must match exactly)
...
</old_str_n>
<new_str_n>
// new code block
...
</new_str_n>

#### **Previous Solutions and Metrics:**

Previously, you have generated the following custom CuTe DSL solutions and got the following runtime metrics:

<Previous Solutions and Metrics>
{previous_kernels_and_metrics}
</Previous Solutions and Metrics>

"""


def _format_experiment_focus(task_params: dict | None) -> str:
    """Return optional experiment-specific tuning guidance."""
    if not task_params:
        return ""
    experiment_focus = task_params.get("experiment_focus")
    if not experiment_focus:
        return ""
    return (
        "\n### Experiment Focus\n\n"
        "When refining the solution, keep this experiment objective in mind "
        "without changing the required definition semantics:\n\n"
        f"{experiment_focus}\n\n"
    )


def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r"\{(\w+)\}", template))


def _is_correct_metric(metric) -> bool:
    """Check if a metric indicates correctness. Handles both EvalResult objects and string representations."""
    if isinstance(metric, EvalResult):
        return metric.correct
    elif isinstance(metric, str):
        # Check string representation for correctness
        metric_lower = metric.lower()
        return (
            "correctness=true" in metric_lower
            or "correct=true" in metric_lower
            or '"correctness": true' in metric_lower
            or '"correct": true' in metric_lower
        )
    return False


def generate_tuner_prompt(
    previous_kernels: list[str] = None,
    previous_metrics: list[str] = None,
    filter_wrong_attempts: bool = False,
    task_params: dict = None,
):
    previous_kernels = previous_kernels or []
    previous_metrics = previous_metrics or []

    # Filter out wrong attempts if requested
    if filter_wrong_attempts:
        filtered_pairs = [
            (kernel, metric)
            for kernel, metric in zip(previous_kernels, previous_metrics)
            if _is_correct_metric(metric)
        ]
        if filtered_pairs:
            previous_kernels, previous_metrics = zip(*filtered_pairs)
            previous_kernels, previous_metrics = list(previous_kernels), list(
                previous_metrics
            )
        else:
            previous_kernels, previous_metrics = [], []

    previous_kernels_and_metrics_str = "\n".join(
        [
            f"\n### {i}-th attempt: \n\n```python\n{kernel}\n```\n\n### {i}-th Runtime Metrics:\n{metric}"
            for i, (kernel, metric) in enumerate(
                zip(previous_kernels, previous_metrics)
            )
        ]
    )

    # Extract required parameters from task prompt template
    required_keys = _extract_format_keys(TASK_INSTRUCTION)

    # Build format dict: use task_params if provided, otherwise fall back to original parameters
    format_dict = {}
    if task_params is not None:
        format_dict.update(task_params)
    format_dict["cute_dsl_scaffold"] = CUTE_DSL_SOURCE_SCAFFOLDING

    # Fall back to original parameters for missing keys
    for key in required_keys:
        if key not in format_dict:
            if key == "previous_kernels_and_metrics":
                format_dict[key] = previous_kernels_and_metrics_str
            else:
                raise ValueError(f"Missing required parameter: {key}")

    prompt = PROBLEM_STATEMENT
    gpu_name = task_params.get("gpu_name")
    gpu_arch = task_params.get("gpu_architecture")
    if gpu_name and gpu_arch:
        prompt += _HARDWARE_INFO.format(gpu_name=gpu_name, gpu_architecture=gpu_arch)
    prompt += TASK_INSTRUCTION.format(**format_dict)
    prompt += _format_experiment_focus(task_params)
    return prompt

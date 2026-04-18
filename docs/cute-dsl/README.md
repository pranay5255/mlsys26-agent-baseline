# CuTe DSL Cheatsheets

This folder summarizes the CuTe DSL details that matter for this agent port. It is
not a replacement for NVIDIA's documentation; it is a repo-local checklist for
prompting, generated solution shape, FlashInfer Bench evaluation, Modal B200
runtime, and common failure modes.

## Source Pages

- Overview: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
- Quick Start: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html
- CuTe DSL introduction: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html
- Control flow: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html
- JIT arguments: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.html
- Dynamic layouts: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html
- Framework integration: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html
- JIT caching: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html
- Compilation options: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html
- Autotuning: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.html
- API reference: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html
- Runtime API: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_runtime.html
- Limitations: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- FAQs: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html

## Files

- `01-installation-runtime.md`: install, Python/CUDA support, uv, Modal B200.
- `02-programming-model.md`: decorators, calling rules, launch shape.
- `03-tensors-layouts-dlpack.md`: PyTorch interop, layouts, dynamic shape.
- `04-control-flow-caching.md`: hybrid DSL behavior and JIT cache patterns.
- `05-api-cheatsheet.md`: high-value API names for generated kernels.
- `06-autotuning.md`: compile cache and tuning guidance.
- `07-debugging-limitations.md`: debugging env vars and unsupported patterns.
- `08-agent-porting-checklist.md`: repo-specific implementation checklist.

## Repo Contract

Generated solutions are Python files evaluated by FlashInfer Bench with:

- language: `python`
- entry point: `main.py::run`
- destination passing style: `False`
- runtime dependency: `nvidia-cutlass-dsl`

The generated `run` function should handle input device movement, output device
restoration, CuTe DSL launch/compile caching, and reference-compatible return
values.

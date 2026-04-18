# Change Log

This file records the per-file changes for the CuTe DSL agent port, DeepSeek DSA
ps1 mini task, Modal B200 smoke coverage, and Python experiment launchers. Each
file committed from the current worktree has a dedicated entry below so commit
messages can point back to a single source of detail.

## changelog.md

- Adds this root change log as the shared line-pointer target for all separate
  file commits.
- Documents the purpose and implementation detail of every file included in the
  commit sequence.

## .codex

- Preserves the empty local `.codex` marker that appears in repository status.
- No runtime behavior is attached to this file.

## .gitignore

- Keeps `uv.lock` trackable even though generic lock files are ignored.
- Narrows dataset tracking to the DSA ps1 definition, workload, and safetensors
  smoke blob needed for the mini task.
- Continues ignoring the larger contest dataset, generated outputs, caches, and
  temporary artifacts.

## README.md

- Updates project documentation from a Triton-oriented baseline to a CuTe DSL
  agent workflow.
- Documents `uv sync`, OpenRouter configuration, Modal B200 execution, local GPU
  execution, and CuTe DSL implementation constraints.

## agent/__init__.py

- Marks `agent` as an importable package for the `pyproject.toml` build target
  and `python -m agent.main` execution.

## agent/api.py

- Adds an `InferenceServer` wrapper that retains provider metadata with the SDK
  client.
- Loads `.env` values without overriding inherited environment variables.
- Adds OpenRouter support through the OpenAI-compatible API using
  `OPENROUTER_API_KEY`, OpenRouter base URL defaults, and optional attribution
  headers.
- Keeps OpenAI and Anthropic provider support while improving missing-credential
  errors.

## agent/eval.py

- Switches generated solution evaluation to Python/CuTe DSL build specs.
- Adds configurable timeout, warmup, iteration, and trial defaults for
  FlashInfer Bench.
- Uses `main.py::run`, `SupportedLanguages.PYTHON`, and
  `nvidia-cutlass-dsl` dependencies for generated solutions.
- Preserves result aggregation into `EvalResult`.

## agent/evolve_agent.py

- Updates terminology from Triton kernels to CuTe DSL solutions where the evolve
  loop reports and stores generated candidates.
- Preserves the existing search and pool mechanics.

## agent/iterative_agent.py

- Updates iterative proposal/refinement flow to request Python/CuTe DSL code
  blocks.
- Passes configured temperature into proposal and tuner LLM calls.
- Preserves existing logging and best-solution selection behavior.

## agent/main.py

- Refactors CLI setup into `build_parser()` and allows `main(argv)` for
  programmatic Python launchers.
- Defaults the workflow to OpenRouter and Modal B200.
- Adds evaluation timing controls and CuTe DSL package version metadata to task
  prompts.
- Adds `--experiment_focus` and passes it into prompt task parameters for
  experiment-specific guidance.
- Syncs the dataset to Modal before running benchmark tasks.

## agent/modal_eval.py

- Builds the Modal runtime from a CUDA 13.1 devel image with Python 3.11.
- Installs `flashinfer-bench`, `torch`, `pydantic`, and
  `nvidia-cutlass-dsl[cu13]`.
- Configures CuTe DSL cache and optional debug logging environment variables.
- Evaluates generated Python/CuTe DSL solutions on Modal B200 with the same
  FlashInfer Bench metadata as local evaluation.
- Resyncs the dataset volume so newly added mini workloads reach Modal.

## agent/utils.py

- Adds `mlsys26-contest` dataset root resolution alongside `flashinfer-trace`.
- Loads task lists that can enumerate explicit problem names by op type.
- Allows YAML config loading to reparse a supplied argv list, enabling Python
  experiment scripts to call the agent without shell subprocesses.

## config/config_evolve.yaml

- Updates the evolve config to target OpenRouter, Modal B200, Blackwell, and
  CuTe DSL evaluation settings.
- Adds benchmark timeout, warmup, iteration, and trial controls.

## config/config_iterative.yaml

- Updates the iterative config to target OpenRouter, Modal B200, Blackwell, and
  CuTe DSL evaluation settings.
- Adds benchmark timeout, warmup, iteration, and trial controls.

## config/config_mini_test.yaml

- Points the mini test configuration at `config/tasks_mini.txt` with Modal B200
  as the evaluation backend.
- Uses OpenRouter defaults and short benchmark counts for fast smoke runs.

## config/tasks_mini.txt

- Replaces the prior mini task with the DeepSeek DSA ps1 sparse-attention task:
  `dsa_paged dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1`.

## datasets/mlsys26-contest/blob/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1_mini_smoke.safetensors

- Adds a tiny safetensors input blob containing a `[1, 2048]` int32
  `sparse_indices` tensor with valid indices for the DSA ps1 mini workload.
- Enables FlashInfer Bench and Modal smoke tests to load non-random sparse
  indices with the expected dtype and shape.

## datasets/mlsys26-contest/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1.json

- Adds the DeepSeek-V3.2 DSA sparse-attention page-size-1 definition.
- Specifies inputs, outputs, axes, constraints, tags, and a PyTorch reference
  implementation for `q_nope`, `q_pe`, compressed KV cache, KPE cache,
  `sparse_indices`, and `sm_scale`.
- Defines output bf16 attention results and fp32 base-2 LSE.

## datasets/mlsys26-contest/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1.jsonl

- Adds a mini workload trace for the DSA ps1 definition.
- Uses one token, 2048 pages, random query/cache tensors, the safetensors sparse
  index blob, and the MLA softmax scale `1/sqrt(192)`.

## docs/cute-dsl/01-installation-runtime.md

- Documents supported Python/CUDA runtime assumptions for CuTe DSL.
- Records the Modal B200 image requirements, dependency set, cache directory,
  and quick import checks.

## docs/cute-dsl/02-programming-model.md

- Summarizes CuTe DSL `@cute.kernel` and `@cute.jit` roles.
- Documents source-file requirements, calling rules, launch parameters, and the
  generated solution shape expected by FlashInfer Bench.

## docs/cute-dsl/03-tensors-layouts-dlpack.md

- Captures PyTorch/DLPack interop, layout construction, dynamic shape handling,
  and tensor conversion guidance for generated CuTe DSL solutions.

## docs/cute-dsl/04-control-flow-caching.md

- Documents CuTe DSL hybrid AST/tracing behavior.
- Lists unsupported dynamic control-flow patterns and module-level executor
  cache key requirements.

## docs/cute-dsl/05-api-cheatsheet.md

- Collects high-value imports, decorators, tensor/layout APIs, DLPack runtime
  helpers, device builtins, sync helpers, copy APIs, and MMA APIs.
- Notes that simpler tensor indexing is preferred for non-GEMM fusion kernels
  unless advanced tiled atoms are built correctly.

## docs/cute-dsl/06-autotuning.md

- Summarizes CuTe DSL autotuning guidance: define search space, compile and
  benchmark each configuration, and cache compiled executors and selected
  configs.
- Maps GEMM tuning concepts to non-GEMM parameters such as block size, vector
  width, elements per thread, and shared-memory staging.

## docs/cute-dsl/07-debugging-limitations.md

- Records CuTe DSL debug environment variables and generated artifact access.
- Documents print behavior, unsupported Python patterns, source inspection
  limitations, and a triage order for generated solution failures.

## docs/cute-dsl/08-agent-porting-checklist.md

- Lists repo-specific requirements for prompts, evaluation, Modal, OpenRouter,
  uv, and smoke commands.
- Acts as a checklist for keeping generated solutions Python/CuTe DSL instead
  of Triton.

## docs/cute-dsl/README.md

- Introduces the local CuTe DSL cheatsheet folder.
- Links the upstream NVIDIA documentation pages used to build the repo-local
  notes.
- States the generated solution contract: Python language, `main.py::run`,
  non-DPS returns, and `nvidia-cutlass-dsl` dependency.

## prompt/__init__.py

- Marks `prompt` as an importable package for the `pyproject.toml` build target
  and tests.

## prompt/proposer_prompt.py

- Rewrites the proposer prompt from Triton generation to Python/CuTe DSL
  solution generation.
- Requires module-scope CuTe decorators, explicit launch parameters, stream
  handling through `cutlass.cuda`, DLPack/layout usage, unsupported-pattern
  avoidance, and module-level `cute.compile` caches.
- Adds optional experiment-focus guidance for fused attention, softmax/LSE, and
  RMSNorm-style reduction runs.

## prompt/tuner_prompt.py

- Rewrites the tuner prompt for Python/CuTe DSL refinement while preserving the
  `str_replace` edit protocol.
- Adds benchmark timing details, CuTe DSL correctness constraints, and
  experiment-focus propagation.
- Improves metric parsing for `correct=True` and JSON-style correctness fields.

## pyproject.toml

- Adds PEP 621 project metadata and hatchling build configuration.
- Declares runtime dependencies including OpenRouter-compatible SDK usage,
  Modal, FlashInfer Bench, Torch, PyYAML, python-dotenv, and
  `nvidia-cutlass-dsl[cu13]`.
- Adds development extras and tooling config for black, isort, and pytest.

## requirements.txt

- Replaces direct dependency management with a note that the project is managed
  through `pyproject.toml` and `uv.lock`.
- Keeps compatibility guidance for users looking at the legacy file.

## scripts/dsa_ps1_experiment_common.py

- Adds shared Python experiment definitions for the three DSA ps1 Modal runs:
  fused attention, softmax/LSE, and RMSNorm-style row reduction.
- Loads `.env`, checks `OPENROUTER_API_KEY`, builds agent argv lists, runs the
  Modal smoke preflight, and invokes `agent.main` programmatically.

## scripts/run_dsa_ps1_attention_fused.py

- Adds a Python launcher for the fused sparse-attention DSA ps1 experiment.
- Uses Modal B200 and focuses the prompt on direct page-size-1 sparse addressing,
  online/chunked reductions, fp32 accumulation, base-2 LSE, and bf16 output.

## scripts/run_dsa_ps1_rmsnorm_reduction.py

- Adds a Python launcher for the RMSNorm-style row-reduction experiment.
- Keeps DSA attention semantics while steering generated code toward reusable
  row-reduction and normalization implementation patterns.

## scripts/run_dsa_ps1_softmax_lse.py

- Adds a Python launcher for the softmax/LSE-focused DSA ps1 experiment.
- Steers generated code toward stable max/sumexp reductions, explicit `-1`
  masking, fp32 reductions, and base-2 LSE correctness.

## scripts/smoke_dsa_definition_modal.py

- Adds a Modal B200 smoke test for the DSA ps1 definition.
- Allocates representative tensors, runs the reference implementation on B200,
  checks output shapes/dtypes/finite values, and compiles/launches a minimal
  CuTe DSL kernel.
- Exposes `run_smoke()` so the Python experiment launchers can reuse the same
  preflight.

## tests/test_cute_port.py

- Adds tests for CuTe prompt requirements, tuner edit protocol, OpenRouter
  client construction, Python/CuTe FlashInfer Bench metadata, experiment-focus
  propagation, programmatic config parsing, and Python experiment argv building.

## uv.lock

- Locks the uv-resolved dependency graph for reproducible local and Modal-facing
  development.

# DSA PS1 Experiment Commands

This repo has three Modal B200 experiment wrappers for the DeepSeek DSA
page-size-1 sparse-attention mini task:

- `scripts/run_dsa_ps1_attention_fused.py`
- `scripts/run_dsa_ps1_softmax_lse.py`
- `scripts/run_dsa_ps1_rmsnorm_reduction.py`

Each wrapper runs `agent.main` with `--eval_backend modal`, `--modal_gpu B200`,
OpenRouter, and `config/tasks_mini.txt`. The wrappers also pass their built-in
experiment objective to the prompts through `--experiment_focus`.

The default fused command can complete successfully while still producing
`Correct: False`. That means the command ran, but the LLM-generated CuTe DSL
candidate failed correctness or runtime checks. The recent failed run produced
runtime errors from unsupported patterns including early `return` inside
`@cute.kernel`, `cute.arch.shared_memory`, direct `cute.load`/`cute.store`, and
`int(...)` on dynamic DSL values. Prefer the compile-first commands below before
running broad search.

## Preconditions

```bash
uv sync
uv run pytest
```

Make sure `.env` or the shell has:

```bash
OPENROUTER_API_KEY=...
```

Run the B200/CuTe smoke once before expensive search:

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py --smoke_only
```

After one smoke pass, add `--skip_smoke` to subsequent commands.

## Command Validation

These commands do not call OpenRouter or Modal evaluation; they only verify that
the wrapper arguments are accepted and that the generated `agent.main` command is
well-formed.

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py --dry_run --total_steps 1 --pool_size 1 --focus_append 'compile-first validation'
uv run python scripts/run_dsa_ps1_softmax_lse.py --dry_run --total_steps 1 --pool_size 1 --focus_append 'compile-first validation'
uv run python scripts/run_dsa_ps1_rmsnorm_reduction.py --dry_run --total_steps 1 --pool_size 1 --focus_append 'compile-first validation'
```

## Shared Prompt Controls

All three wrappers accept these prompt-specific controls:

- `--focus_append '...'`: append extra guidance to the built-in experiment
  focus. Can be passed multiple times.
- `--focus_override '...'`: replace the built-in focus completely.

Use `--focus_append` for most runs because it preserves the task-specific focus
while adding lessons from failed candidates.

Common compile-first guidance:

```bash
--focus_append 'Compile-first constraints: do not use cute.load, cute.store, cute.shared_memory, cute.arch.shared_memory, bracket launch syntax, early return inside @cute.kernel, or int(...) on dynamic CuTe DSL values. Use module-scope @cute.kernel and @cute.jit, cute.arch thread/block builtins, from_dlpack only from cutlass.cute.runtime, cutlass.cuda.default_stream(), cutlass.cuda.stream_sync(stream), and kernel(args...).launch(grid=..., block=..., stream=...).'
```

## Fused Sparse Attention

Use this when exploring one-kernel or minimal-kernel sparse attention. Start with
the compile-first variant; the plain default is intentionally broad and can
waste steps on runtime errors.

Cheap compile-first run:

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py \
  --skip_smoke \
  --total_steps 3 \
  --pool_size 2 \
  --temperature 0.35 \
  --eval_timeout 300 \
  --eval_warmup_runs 2 \
  --eval_iterations 3 \
  --eval_num_trials 1 \
  --max_completion_tokens 16000 \
  --save_path outputs/experiments/dsa_ps1_attention_fused_compile_first \
  --focus_append 'Compile-first constraints: do not use cute.load, cute.store, cute.shared_memory, cute.arch.shared_memory, bracket launch syntax, early return inside @cute.kernel, or int(...) on dynamic CuTe DSL values. If a boundary check is needed, predicate work instead of returning. Prefer a simple valid CuTe DSL kernel over aggressive shared-memory tiling.'
```

Runtime-debug run with CuTe logs:

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py \
  --skip_smoke \
  --debug \
  --total_steps 2 \
  --pool_size 1 \
  --temperature 0.30 \
  --eval_timeout 360 \
  --eval_warmup_runs 2 \
  --eval_iterations 2 \
  --eval_num_trials 1 \
  --save_path outputs/experiments/dsa_ps1_attention_fused_debug \
  --focus_append 'Debug runtime failures. Avoid early return in @cute.kernel, avoid cute.arch.shared_memory, avoid Python int() conversion of DSL values, and keep dynamic tensor values in CuTe scalar form.'
```

Broader fused exploration after at least one candidate compiles:

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py \
  --skip_smoke \
  --total_steps 8 \
  --pool_size 5 \
  --temperature 0.50 \
  --eval_timeout 360 \
  --eval_warmup_runs 3 \
  --eval_iterations 5 \
  --eval_num_trials 1 \
  --max_completion_tokens 22000 \
  --save_path outputs/experiments/dsa_ps1_attention_fused_explore \
  --focus_append 'Preserve compile-first constraints from the failed run. Explore online or chunked softmax and value accumulation only after the CuTe launch, tensor conversion, and kernel control flow are valid.'
```

Deep fused benchmark run:

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py \
  --skip_smoke \
  --total_steps 14 \
  --pool_size 8 \
  --temperature 0.45 \
  --eval_timeout 480 \
  --eval_warmup_runs 5 \
  --eval_iterations 10 \
  --eval_num_trials 3 \
  --max_completion_tokens 24000 \
  --save_path outputs/experiments/dsa_ps1_attention_fused_deep \
  --focus_append 'Benchmark-stability run. Reuse module-level cute.compile caches in timed iterations and avoid new APIs unless already validated by a smaller run.'
```

## Softmax And LSE

Use this when isolating stable max/sumexp and base-2 LSE behavior.

Cheap compile-first softmax run:

```bash
uv run python scripts/run_dsa_ps1_softmax_lse.py \
  --skip_smoke \
  --total_steps 3 \
  --pool_size 2 \
  --temperature 0.45 \
  --eval_timeout 300 \
  --eval_warmup_runs 2 \
  --eval_iterations 3 \
  --eval_num_trials 1 \
  --max_completion_tokens 16000 \
  --save_path outputs/experiments/dsa_ps1_softmax_lse_compile_first \
  --focus_append 'Compile-first softmax/LSE. Prefer a simple two-pass max then sumexp implementation over aggressive fusion. Avoid cute.load, cute.store, shared-memory helper guesses, early return inside @cute.kernel, and int(...) on dynamic DSL values.'
```

Online softmax exploration:

```bash
uv run python scripts/run_dsa_ps1_softmax_lse.py \
  --skip_smoke \
  --total_steps 8 \
  --pool_size 4 \
  --temperature 0.65 \
  --eval_timeout 360 \
  --eval_warmup_runs 3 \
  --eval_iterations 5 \
  --eval_num_trials 1 \
  --max_completion_tokens 22000 \
  --save_path outputs/experiments/dsa_ps1_softmax_lse_online \
  --focus_append 'Explore online softmax only with valid CuTe DSL control flow. Maintain running max and denominator in fp32, mask sparse_indices == -1 before max and denominator updates, and store lse as logsumexp / log(2).'
```

Deep softmax/LSE benchmark run:

```bash
uv run python scripts/run_dsa_ps1_softmax_lse.py \
  --skip_smoke \
  --total_steps 12 \
  --pool_size 5 \
  --temperature 0.50 \
  --eval_timeout 420 \
  --eval_warmup_runs 5 \
  --eval_iterations 10 \
  --eval_num_trials 3 \
  --max_completion_tokens 24000 \
  --save_path outputs/experiments/dsa_ps1_softmax_lse_deep \
  --focus_append 'Benchmark stable candidates. Keep exact DSA semantics: logits are q_nope dot ckv plus q_pe dot kpe, scaled by sm_scale; output is softmax(logits_scaled) @ ckv; lse is fp32 base 2.'
```

## RMSNorm-Style Row Reduction

Use this to bias the attention implementation toward reusable row-reduction and
normalization structure without changing the DSA math.

Cheap compile-first reduction run:

```bash
uv run python scripts/run_dsa_ps1_rmsnorm_reduction.py \
  --skip_smoke \
  --total_steps 3 \
  --pool_size 2 \
  --temperature 0.65 \
  --eval_timeout 300 \
  --eval_warmup_runs 2 \
  --eval_iterations 3 \
  --eval_num_trials 1 \
  --max_completion_tokens 16000 \
  --save_path outputs/experiments/dsa_ps1_rmsnorm_reduction_compile_first \
  --focus_append 'Compile-first row reduction. Do not implement RMSNorm. Use the RMSNorm analogy only for stable per-row fp32 reductions and reciprocal normalization. Avoid unsupported shared-memory helpers, early return, and Python int conversion of dynamic DSL values.'
```

Reduction-pattern exploration:

```bash
uv run python scripts/run_dsa_ps1_rmsnorm_reduction.py \
  --skip_smoke \
  --total_steps 8 \
  --pool_size 5 \
  --temperature 0.80 \
  --eval_timeout 360 \
  --eval_warmup_runs 3 \
  --eval_iterations 5 \
  --eval_num_trials 1 \
  --max_completion_tokens 22000 \
  --save_path outputs/experiments/dsa_ps1_rmsnorm_reduction_explore \
  --focus_append 'Explore deterministic reduction trees and reciprocal normalization patterns while preserving sparse attention semantics and base-2 LSE output.'
```

Deep reduction benchmark run:

```bash
uv run python scripts/run_dsa_ps1_rmsnorm_reduction.py \
  --skip_smoke \
  --total_steps 14 \
  --pool_size 8 \
  --temperature 0.75 \
  --eval_timeout 480 \
  --eval_warmup_runs 5 \
  --eval_iterations 10 \
  --eval_num_trials 3 \
  --max_completion_tokens 24000 \
  --save_path outputs/experiments/dsa_ps1_rmsnorm_reduction_deep \
  --focus_append 'Benchmark stable reduction candidates only. Keep exact attention output and lse semantics; do not replace the task with RMSNorm.'
```

## Focus Override Examples

Use `--focus_override` when you want a completely different prompt objective.
This is useful after repeated failures from a broad built-in focus.

Minimal valid CuTe probe objective:

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py \
  --skip_smoke \
  --total_steps 2 \
  --pool_size 1 \
  --temperature 0.25 \
  --save_path outputs/experiments/dsa_ps1_attention_minimal_valid \
  --focus_override 'Generate the simplest valid CuTe DSL solution for the DSA ps1 task. Prioritize compiling and returning correctly shaped tensors over speed. Avoid cute.load, cute.store, cute.shared_memory, cute.arch.shared_memory, early return inside @cute.kernel, and int(...) on dynamic DSL values. Use the documented scaffold exactly for imports, stream, compile cache, and launch syntax.'
```

## Resume Commands

Resume a partially useful run while increasing the step budget:

```bash
uv run python scripts/run_dsa_ps1_attention_fused.py \
  --skip_smoke \
  --resume_from outputs/experiments/dsa_ps1_attention_fused_compile_first \
  --total_steps 8 \
  --pool_size 5 \
  --temperature 0.45 \
  --save_path outputs/experiments/dsa_ps1_attention_fused_resume_8 \
  --focus_append 'Continue from prior attempts. Avoid repeating any runtime-error pattern reported in previous metrics.'
```

## Reading Results

After a run, inspect metrics before launching a deeper run:

```bash
python -m json.tool outputs/experiments/dsa_ps1_attention_fused_compile_first/dsa_paged_dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1/proposal_1_metrics.json
```

Useful failure meanings:

- `STATIC_CUTE_SOURCE_ERROR`: the local preflight rejected known bad CuTe DSL
  patterns before spending B200 compile time.
- `COMPILE_ERROR`: FlashInfer Bench could not import or build the generated
  Python/CuTe solution.
- `RUNTIME_ERROR`: the solution imported, but failed during correctness,
  compilation through `cute.compile`, or kernel execution.
- `Correct: False` with completed app: the infrastructure worked; the generated
  candidate was not yet correct.

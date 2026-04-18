"""
CuTe DSL solution evaluation using flashinfer-bench Python API directly.
"""

import functools
import json
import logging
import os
import uuid

from agent.diagnostics import build_eval_diagnostics
from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    BuildSpec,
    EvaluationStatus,
    Solution,
    SourceFile,
    SupportedLanguages,
    TraceSet,
)
from pydantic import BaseModel

from agent.utils import prepare_cute_source_for_eval

logger = logging.getLogger(__name__)

DEFAULT_EVAL_TIMEOUT_SECONDS = 180
DEFAULT_WARMUP_RUNS = 3
DEFAULT_BENCHMARK_ITERATIONS = 5
DEFAULT_BENCHMARK_TRIALS = 1
CUTE_DSL_DEPENDENCIES = ["nvidia-cutlass-dsl"]


class EvalResult(BaseModel):
    """Result of evaluating a single generated solution."""

    compiled: bool = False
    correct: bool = False
    speedup: float = 0.0
    latency_ms: float | None = None
    task_id: str = ""
    error: str | None = None
    stats: dict | None = None
    diagnostics: dict | None = None


def _attach_diagnostics(
    result: EvalResult, source_issues: list[str] | None = None
) -> EvalResult:
    if result.diagnostics is None:
        result.diagnostics = build_eval_diagnostics(
            compiled=result.compiled,
            correct=result.correct,
            error=result.error,
            stats=result.stats,
            source_issues=source_issues,
        )
    return result


def _configure_cute_deep_logs() -> None:
    """Enable detailed CuTe DSL host-side compilation logs by default."""
    os.environ.setdefault("CUTE_DSL_LINEINFO", "1")
    os.environ.setdefault("CUTE_DSL_LOG_TO_CONSOLE", "1")
    os.environ.setdefault("CUTE_DSL_LOG_LEVEL", "20")


def _summarize_traces(traces) -> list[dict]:
    summaries = []
    for idx, trace in enumerate(traces):
        ev = getattr(trace, "evaluation", None)
        if ev is None:
            summaries.append({"index": idx, "status": "NO_EVALUATION"})
            continue
        status = getattr(getattr(ev, "status", None), "value", None) or str(
            getattr(ev, "status", "unknown")
        )
        log = getattr(ev, "log", None) or ""
        entry = {
            "index": idx,
            "status": status,
            "log_length": len(log),
            "log_tail": log[-6000:],
        }
        performance = getattr(ev, "performance", None)
        if performance is not None:
            entry["performance"] = {
                "latency_ms": getattr(performance, "latency_ms", None),
                "reference_latency_ms": getattr(
                    performance, "reference_latency_ms", None
                ),
                "speedup_factor": getattr(performance, "speedup_factor", None),
            }
        correctness = getattr(ev, "correctness", None)
        if correctness is not None:
            entry["correctness"] = {
                "max_relative_error": getattr(correctness, "max_relative_error", None),
                "max_absolute_error": getattr(correctness, "max_absolute_error", None),
            }
        summaries.append(entry)
    return summaries


def calculate_score(metric: EvalResult):
    """Return (compiled, correct, speedup) tuple for ranking."""
    if metric is None:
        return (0, 0, 0)
    if not metric.compiled:
        return (0, 0, 0)
    if not metric.correct:
        return (1, 0, 0)
    return (1, 1, metric.speedup)


def read_metrics(metrics_path: str, full: bool = False):
    """
    Read metrics from a JSON file.

    Returns:
        If full: EvalResult object
        Otherwise: tuple (correct: bool, speedup: float)
    """
    with open(metrics_path, "r") as f:
        data = json.load(f)

    if full:
        return EvalResult(**data)

    if data.get("compiled") and data.get("correct"):
        return (True, data.get("speedup", 0.0))
    return (False, 0.0)


def create_eval_fn(
    backend: str = "local",
    dataset_name: str = "mlsys26-contest",
    remote_fn=None,
    timeout: int = DEFAULT_EVAL_TIMEOUT_SECONDS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    num_trials: int = DEFAULT_BENCHMARK_TRIALS,
):
    """Factory to create eval function based on backend.

    Args:
        backend: "local" for local GPU, "modal" for Modal remote GPU.
        dataset_name: Dataset subdirectory name (used by modal backend).
        remote_fn: Modal remote function (required when backend="modal").

    Returns:
        Callable with same signature as eval_kernel.
    """
    if backend == "local":
        return functools.partial(
            eval_kernel,
            timeout=timeout,
            warmup_runs=warmup_runs,
            iterations=iterations,
            num_trials=num_trials,
        )
    elif backend == "modal":
        if remote_fn is None:
            raise ValueError("remote_fn is required for modal backend")

        def _modal_eval(kernel_code, task_id, dataset_root):
            kernel_code, source_issues = prepare_cute_source_for_eval(kernel_code)
            if source_issues:
                return _attach_diagnostics(
                    EvalResult(
                        compiled=False,
                        task_id=task_id,
                        error="STATIC_CUTE_SOURCE_ERROR: " + " | ".join(source_issues),
                    ),
                    source_issues=source_issues,
                )
            result_dict = remote_fn.remote(
                kernel_code,
                task_id,
                dataset_name,
                timeout,
                warmup_runs,
                iterations,
                num_trials,
            )
            return _attach_diagnostics(EvalResult(**result_dict))

        return _modal_eval
    else:
        raise ValueError(f"Unknown eval backend: {backend}")


def eval_kernel(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
    timeout: int = DEFAULT_EVAL_TIMEOUT_SECONDS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    num_trials: int = DEFAULT_BENCHMARK_TRIALS,
) -> EvalResult:
    """
    Evaluate a CuTe DSL Python solution against the reference.

    Args:
        kernel_code: Source code of the solution to evaluate.
        task_id: Definition/problem name (e.g. "moe_fp8_block_scale_...").
        dataset_root: Path to the dataset root directory.
        timeout: Timeout in seconds per solution evaluation.

    Returns:
        EvalResult with compiled, correct, speedup, latency_ms, etc.
    """
    _configure_cute_deep_logs()
    kernel_code, source_issues = prepare_cute_source_for_eval(kernel_code)
    if source_issues:
        return _attach_diagnostics(
            EvalResult(
                compiled=False,
                task_id=task_id,
                error="STATIC_CUTE_SOURCE_ERROR: " + " | ".join(source_issues),
            ),
            source_issues=source_issues,
        )
    trace_set = TraceSet.from_path(dataset_root)

    solution_name = f"agent_{uuid.uuid4().hex[:8]}"

    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="agent",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            dependencies=CUTE_DSL_DEPENDENCIES,
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=kernel_code)],
    )

    # Inject solution into in-memory trace set
    trace_set.solutions.setdefault(task_id, []).append(solution)
    trace_set._solution_by_name[solution_name] = solution

    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
        definitions=[task_id],
        solutions=[solution_name],
        timeout_seconds=timeout,
    )

    benchmark = Benchmark(trace_set, config)
    try:
        result_ts = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()

    traces = result_ts.traces.get(task_id, [])

    # Find first error
    error_statuses = {
        EvaluationStatus.COMPILE_ERROR,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_NUMERICAL,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.TIMEOUT,
    }
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status in error_statuses:
            return _attach_diagnostics(
                EvalResult(
                    compiled=(ev.status != EvaluationStatus.COMPILE_ERROR),
                    task_id=task_id,
                    error=f"{ev.status.value}: {ev.log}",
                    stats={"trace_evaluations": _summarize_traces(traces)},
                )
            )

    # Aggregate PASSED results
    latencies, ref_latencies, speedups = [], [], []
    rel_errors, abs_errors = [], []
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status == EvaluationStatus.PASSED:
            latencies.append(ev.performance.latency_ms)
            ref_latencies.append(ev.performance.reference_latency_ms)
            speedups.append(ev.performance.speedup_factor)
            rel_errors.append(ev.correctness.max_relative_error)
            abs_errors.append(ev.correctness.max_absolute_error)

    if not latencies:
        return _attach_diagnostics(
            EvalResult(
                task_id=task_id,
                error="No evaluation results",
                stats={"trace_evaluations": _summarize_traces(traces)},
            )
        )

    n = len(latencies)
    avg_speedup = sum(speedups) / n
    return _attach_diagnostics(
        EvalResult(
            compiled=True,
            correct=True,
            speedup=avg_speedup,
            latency_ms=sum(latencies) / n,
            task_id=task_id,
            stats={
                "reference_latency_ms": sum(ref_latencies) / n,
                "max_relative_error": sum(rel_errors) / n,
                "max_absolute_error": sum(abs_errors) / n,
                "total_workloads": n,
                "trace_evaluations": _summarize_traces(traces),
            },
        )
    )

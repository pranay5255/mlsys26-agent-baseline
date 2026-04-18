"""
Remote CuTe DSL solution evaluation on Modal GPU.

Usage: set eval_backend="modal" in config or --eval_backend modal on CLI.
The Modal app is started programmatically via app.run() in main.py,
no `modal deploy` or `modal run` CLI commands needed.
"""

import logging
import os

import modal

logger = logging.getLogger(__name__)

MODAL_APP_NAME = "flashinfer-bench-agent"
VOLUME_NAME = "flashinfer-trace"
DATASET_PATH = "/data"
CUTE_CACHE_PATH = "/tmp/cute_dsl_cache"


def create_modal_app(gpu_type: str = "B200", debug: bool = False):
    """Create Modal app, remote eval function, and dataset volume.

    Args:
        gpu_type: GPU type string for Modal (e.g. "B200", "A100", "H100").

    Returns:
        Tuple of (app, remote_eval_kernel, dataset_vol).
    """
    app = modal.App(MODAL_APP_NAME)

    env = {"CUTE_DSL_CACHE_DIR": CUTE_CACHE_PATH}
    if debug:
        env.update(
            {
                "CUTE_DSL_LINEINFO": "1",
                "CUTE_DSL_LOG_TO_CONSOLE": "1",
                "CUTE_DSL_LOG_LEVEL": "20",
            }
        )

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.11"
        )
        .pip_install(
            "flashinfer-bench",
            "torch",
            "pydantic",
            "nvidia-cutlass-dsl[cu13]",
        )
        .env(env)
    )

    dataset_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        gpu=gpu_type,
        volumes={DATASET_PATH: dataset_vol},
        timeout=300,
        serialized=True,
    )
    def remote_eval_kernel(
        kernel_code: str,
        task_id: str,
        dataset_name: str,
        timeout: int = 180,
        warmup_runs: int = 3,
        iterations: int = 5,
        num_trials: int = 1,
    ) -> dict:
        """Evaluate a CuTe DSL solution on Modal GPU. Mirrors eval.eval_kernel()."""
        import importlib.metadata
        import re
        import uuid

        replacements = {
            "from cutlass.cute import from_dlpack": (
                "from cutlass.cute.runtime import from_dlpack"
            ),
            "from cutlass import from_dlpack": (
                "from cutlass.cute.runtime import from_dlpack"
            ),
            "from cutlass.cuda import default_stream": "",
            "from cutlass.cute.cuda import default_stream": "",
            "stream.synchronize()": "cutlass.cuda.stream_sync(stream)",
        }
        for old, new in replacements.items():
            kernel_code = kernel_code.replace(old, new)
        kernel_code = re.sub(
            r"(?<![\w.])default_stream\(\)",
            "cutlass.cuda.default_stream()",
            kernel_code,
        )
        lines = []
        seen_runtime_dlpack_import = False
        for line in kernel_code.splitlines():
            if line == "from cutlass.cute.runtime import from_dlpack":
                if seen_runtime_dlpack_import:
                    continue
                seen_runtime_dlpack_import = True
            lines.append(line)
        kernel_code = "\n".join(lines)
        has_cutlass_import = any(
            line == "import cutlass" or line.startswith("import cutlass ")
            for line in lines
        )
        if "cutlass.cuda." in kernel_code and not has_cutlass_import:
            kernel_code = "import cutlass\n" + kernel_code

        try:
            import cutlass  # noqa: F401
            import cutlass.cute as cute  # noqa: F401
            import torch  # noqa: F401
        except Exception as e:
            return {
                "compiled": False,
                "task_id": task_id,
                "error": f"CuTe DSL import failed on Modal image: {type(e).__name__}: {e}",
            }

        from flashinfer_bench.bench import Benchmark, BenchmarkConfig
        from flashinfer_bench.data import (
            BuildSpec,
            EvaluationStatus,
            Solution,
            SourceFile,
            SupportedLanguages,
            TraceSet,
        )

        dataset_root = os.path.join(DATASET_PATH, dataset_name)
        trace_set = TraceSet.from_path(dataset_root)

        solution_name = f"agent_{uuid.uuid4().hex[:8]}"
        try:
            cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
        except importlib.metadata.PackageNotFoundError:
            cutlass_version = "unknown"

        solution = Solution(
            name=solution_name,
            definition=task_id,
            author="agent",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="main.py::run",
                dependencies=["nvidia-cutlass-dsl"],
                destination_passing_style=False,
            ),
            sources=[
                SourceFile(
                    path="main.py",
                    content=(
                        f"# Evaluated on Modal {gpu_type} with nvidia-cutlass-dsl "
                        f"{cutlass_version}\n" + kernel_code
                    ),
                )
            ],
        )

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
                return {
                    "compiled": ev.status != EvaluationStatus.COMPILE_ERROR,
                    "task_id": task_id,
                    "error": f"{ev.status.value}: {ev.log}",
                }

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
            return {"task_id": task_id, "error": "No evaluation results"}

        n = len(latencies)
        return {
            "compiled": True,
            "correct": True,
            "speedup": sum(speedups) / n,
            "latency_ms": sum(latencies) / n,
            "task_id": task_id,
            "stats": {
                "reference_latency_ms": sum(ref_latencies) / n,
                "max_relative_error": sum(rel_errors) / n,
                "max_absolute_error": sum(abs_errors) / n,
                "total_workloads": n,
            },
        }

    return app, remote_eval_kernel, dataset_vol


def ensure_dataset_synced(
    dataset_vol: modal.Volume, local_dataset_root: str, dataset_name: str
):
    """Ensure benchmark dataset payload is uploaded to Modal Volume."""
    if not os.path.isdir(local_dataset_root):
        raise FileNotFoundError(
            f"Local dataset root does not exist or is not a directory: {local_dataset_root}"
        )

    logger.info(f"Syncing dataset '{dataset_name}' to Modal Volume...")
    with dataset_vol.batch_upload(force=True) as batch:
        for dirname in ("definitions", "workloads", "blob", "solutions"):
            local_path = os.path.join(local_dataset_root, dirname)
            if os.path.isdir(local_path):
                batch.put_directory(local_path, f"/{dataset_name}/{dirname}")

        readme_path = os.path.join(local_dataset_root, "README.md")
        if os.path.isfile(readme_path):
            batch.put_file(readme_path, f"/{dataset_name}/README.md")
    logger.info(f"Dataset synced to Modal Volume '{VOLUME_NAME}'")

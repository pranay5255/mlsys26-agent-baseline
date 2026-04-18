from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tarfile
import textwrap
import uuid
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.modal_eval import (
    DATASET_PATH,
    VOLUME_NAME,
    ensure_dataset_synced,
)  # noqa: E402
from agent.utils import get_dataset_root, prepare_cute_source_for_eval  # noqa: E402

APP_NAME = "flashinfer-bench-agent-profile"
PROFILE_ROOT = "/_profiles"
DEFAULT_TASK_ID = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1"


RUNNER_SOURCE = r"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import traceback
import uuid

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    BuildSpec,
    EvaluationStatus,
    Solution,
    SourceFile,
    SupportedLanguages,
    TraceSet,
)


def _status_value(status):
    return getattr(status, "value", None) or str(status)


def _summarize_traces(result_ts, task_id):
    traces = result_ts.traces.get(task_id, [])
    summaries = []
    for idx, trace in enumerate(traces):
        ev = getattr(trace, "evaluation", None)
        if ev is None:
            summaries.append({"index": idx, "status": "NO_EVALUATION"})
            continue
        log = getattr(ev, "log", None) or ""
        entry = {
            "index": idx,
            "status": _status_value(getattr(ev, "status", "unknown")),
            "log_length": len(log),
            "log_tail": log[-12000:],
        }
        perf = getattr(ev, "performance", None)
        if perf is not None:
            entry["performance"] = {
                "latency_ms": getattr(perf, "latency_ms", None),
                "reference_latency_ms": getattr(perf, "reference_latency_ms", None),
                "speedup_factor": getattr(perf, "speedup_factor", None),
            }
        correctness = getattr(ev, "correctness", None)
        if correctness is not None:
            entry["correctness"] = {
                "max_relative_error": getattr(correctness, "max_relative_error", None),
                "max_absolute_error": getattr(correctness, "max_absolute_error", None),
            }
        summaries.append(entry)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-file", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dump-traces", action="store_true")
    args = parser.parse_args()

    kernel_code = open(args.kernel_file, "r", encoding="utf-8").read()
    solution_name = f"profile_{uuid.uuid4().hex[:8]}"
    try:
        cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        cutlass_version = "unknown"

    result = {
        "ok": False,
        "solution_name": solution_name,
        "task_id": args.task_id,
        "nvidia_cutlass_dsl": cutlass_version,
    }
    benchmark = None
    try:
        trace_set = TraceSet.from_path(args.dataset_root)
        solution = Solution(
            name=solution_name,
            definition=args.task_id,
            author="agent-profile",
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
                        "# Profiled on Modal B200 with nvidia-cutlass-dsl "
                        f"{cutlass_version}\n" + kernel_code
                    ),
                )
            ],
        )
        trace_set.solutions.setdefault(args.task_id, []).append(solution)
        trace_set._solution_by_name[solution_name] = solution
        config = BenchmarkConfig(
            warmup_runs=args.warmup_runs,
            iterations=args.iterations,
            num_trials=args.num_trials,
            definitions=[args.task_id],
            solutions=[solution_name],
            timeout_seconds=args.timeout_seconds,
        )
        benchmark = Benchmark(trace_set, config)
        result_ts = benchmark.run_all(dump_traces=args.dump_traces)
        traces = _summarize_traces(result_ts, args.task_id)
        result.update({"ok": True, "trace_evaluations": traces})
        statuses = [trace.get("status") for trace in traces]
        result["statuses"] = statuses
        result["passed"] = any(status == EvaluationStatus.PASSED.value for status in statuses)
    except BaseException as exc:
        result.update(
            {
                "ok": False,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if benchmark is not None:
            benchmark.close()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a saved CuTe DSL solution under NCU/NSYS on Modal B200."
    )
    parser.add_argument("--kernel_path", required=True)
    parser.add_argument("--task_id", default=DEFAULT_TASK_ID)
    parser.add_argument("--dataset_name", default="mlsys26-contest")
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--output_dir", default="outputs/profiles")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--eval_timeout", type=int, default=480)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument(
        "--profiler",
        choices=["ncu", "nsys", "none"],
        default="ncu",
        help="Profiler to wrap around the benchmark process.",
    )
    parser.add_argument("--ncu_set", default="full")
    parser.add_argument("--ncu_metrics", default=None)
    parser.add_argument("--ncu_kernel_name", default=None)
    parser.add_argument("--ncu_launch_skip", type=int, default=None)
    parser.add_argument("--ncu_launch_count", type=int, default=None)
    parser.add_argument("--dump_traces", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def create_profile_app(gpu_type: str):
    app = modal.App(APP_NAME)
    env = {
        "CUTE_DSL_CACHE_DIR": "/tmp/cute_dsl_cache",
        "CUTE_DSL_LINEINFO": "1",
        "CUTE_DSL_LOG_TO_CONSOLE": "1",
        "CUTE_DSL_LOG_LEVEL": "20",
    }
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
    profile_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        gpu=gpu_type,
        volumes={DATASET_PATH: profile_vol},
        timeout=1200,
        serialized=True,
    )
    def remote_profile(
        kernel_code: str,
        task_id: str,
        dataset_name: str,
        run_id: str,
        profiler: str,
        ncu_set: str,
        ncu_metrics: str | None,
        ncu_kernel_name: str | None,
        ncu_launch_skip: int | None,
        ncu_launch_count: int | None,
        warmup_runs: int,
        iterations: int,
        num_trials: int,
        eval_timeout: int,
        subprocess_timeout: int,
        dump_traces: bool,
    ) -> dict:
        import os
        import shutil
        import subprocess
        import sys
        import tarfile
        import time
        from pathlib import Path

        output_dir = Path(DATASET_PATH) / PROFILE_ROOT.strip("/") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        kernel_file = output_dir / "kernel_under_test.py"
        runner_file = output_dir / "profile_runner.py"
        benchmark_json = output_dir / "benchmark_result.json"
        command_json = output_dir / "profiler_command.json"
        stdout_file = output_dir / "profiler_stdout.txt"
        stderr_file = output_dir / "profiler_stderr.txt"

        kernel_file.write_text(kernel_code, encoding="utf-8")
        runner_file.write_text(RUNNER_SOURCE, encoding="utf-8")

        ncu_path = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
        if not os.path.exists(ncu_path):
            ncu_path = None
        nsys_path = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
        if not os.path.exists(nsys_path):
            nsys_path = None

        python_cmd = [
            sys.executable,
            str(runner_file),
            "--kernel-file",
            str(kernel_file),
            "--task-id",
            task_id,
            "--dataset-root",
            str(Path(DATASET_PATH) / dataset_name),
            "--warmup-runs",
            str(warmup_runs),
            "--iterations",
            str(iterations),
            "--num-trials",
            str(num_trials),
            "--timeout-seconds",
            str(eval_timeout),
            "--output-json",
            str(benchmark_json),
        ]
        if dump_traces:
            python_cmd.append("--dump-traces")

        if profiler == "ncu":
            if ncu_path is None:
                raise RuntimeError("ncu was not found in the Modal CUDA image")
            report_base = output_dir / "ncu_full_report"
            cmd = [
                ncu_path,
                "--set",
                ncu_set,
                "--target-processes",
                "all",
                "--print-summary",
                "per-kernel",
                "--force-overwrite",
                "--export",
                str(report_base),
                "--log-file",
                str(output_dir / "ncu.log"),
            ]
            if ncu_metrics:
                cmd.extend(["--metrics", ncu_metrics])
            if ncu_kernel_name:
                cmd.extend(["--kernel-name", ncu_kernel_name])
            if ncu_launch_skip is not None:
                cmd.extend(["--launch-skip", str(ncu_launch_skip)])
            if ncu_launch_count is not None:
                cmd.extend(["--launch-count", str(ncu_launch_count)])
            cmd.extend(python_cmd)
        elif profiler == "nsys":
            if nsys_path is None:
                raise RuntimeError("nsys was not found in the Modal CUDA image")
            cmd = [
                nsys_path,
                "profile",
                "--force-overwrite=true",
                "--trace=cuda,nvtx,osrt",
                "-o",
                str(output_dir / "nsys_report"),
            ] + python_cmd
        else:
            cmd = python_cmd

        command_payload = {
            "run_id": run_id,
            "profiler": profiler,
            "command": cmd,
            "ncu_path": ncu_path,
            "nsys_path": nsys_path,
            "created_unix": time.time(),
        }
        command_json.write_text(json.dumps(command_payload, indent=2), encoding="utf-8")

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=subprocess_timeout,
            )
            stdout_file.write_text(completed.stdout, encoding="utf-8")
            stderr_file.write_text(completed.stderr, encoding="utf-8")
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout_file.write_text(exc.stdout or "", encoding="utf-8")
            stderr_file.write_text(exc.stderr or "", encoding="utf-8")
            (output_dir / "profiler_timeout.txt").write_text(str(exc), encoding="utf-8")
            returncode = 124
            timed_out = True

        manifest = []
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                manifest.append(
                    {
                        "path": str(path.relative_to(output_dir)),
                        "bytes": path.stat().st_size,
                    }
                )
        (output_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        tar_path = output_dir / "profile_artifacts.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for path in output_dir.rglob("*"):
                if path == tar_path:
                    continue
                tar.add(path, arcname=path.relative_to(output_dir))

        profile_vol.commit()
        return {
            "run_id": run_id,
            "returncode": returncode,
            "timed_out": timed_out,
            "profiler": profiler,
            "remote_dir": f"{PROFILE_ROOT}/{run_id}",
            "remote_tar": f"{PROFILE_ROOT}/{run_id}/profile_artifacts.tar.gz",
            "manifest": manifest,
            "benchmark_json_exists": benchmark_json.exists(),
        }

    return app, remote_profile, profile_vol


def _download_profile_artifacts(volume, remote_tar: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    tar_path = local_dir / "profile_artifacts.tar.gz"
    volume.reload()
    with tar_path.open("wb") as f:
        volume.read_file_into_fileobj(remote_tar, f)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(local_dir)
    return tar_path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    kernel_path = Path(args.kernel_path).expanduser().resolve()
    if not kernel_path.exists():
        raise FileNotFoundError(kernel_path)

    kernel_code = kernel_path.read_text(encoding="utf-8")
    kernel_code, source_issues = prepare_cute_source_for_eval(kernel_code)
    if source_issues:
        print("Static CuTe source issues were found before profiling:")
        for issue in source_issues:
            print(f"- {issue}")
        print("No GPU kernel will launch until these are fixed.")
        if args.profiler != "none":
            raise SystemExit(2)

    run_name = args.run_name or f"{kernel_path.stem}_{uuid.uuid4().hex[:8]}"
    local_dir = Path(args.output_dir).expanduser().resolve() / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    display_cmd = [
        "uv",
        "run",
        "python",
        "scripts/profile_modal_ncu.py",
        "--kernel_path",
        str(kernel_path),
        "--task_id",
        args.task_id,
        "--profiler",
        args.profiler,
        "--output_dir",
        str(Path(args.output_dir).expanduser()),
        "--run_name",
        run_name,
    ]
    print("Profile command:")
    print(" ".join(shlex.quote(part) for part in display_cmd))
    print(f"Local output directory: {local_dir}")
    if args.dry_run:
        return

    app, remote_profile, profile_vol = create_profile_app(args.gpu)
    with modal.enable_output(), app.run():
        ensure_dataset_synced(
            profile_vol, get_dataset_root(args.dataset_name), args.dataset_name
        )
        result = remote_profile.remote(
            kernel_code,
            args.task_id,
            args.dataset_name,
            run_name,
            args.profiler,
            args.ncu_set,
            args.ncu_metrics,
            args.ncu_kernel_name,
            args.ncu_launch_skip,
            args.ncu_launch_count,
            args.warmup_runs,
            args.iterations,
            args.num_trials,
            args.eval_timeout,
            args.timeout,
            args.dump_traces,
        )
        (local_dir / "remote_profile_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        tar_path = _download_profile_artifacts(
            profile_vol, result["remote_tar"], local_dir
        )

    print(f"Downloaded profile artifacts: {tar_path}")
    print(f"Remote return code: {result['returncode']}")
    print(f"Remote volume path: {result['remote_dir']}")


if __name__ == "__main__":
    main()

"""Structured diagnostics for generated CuTe DSL solution evaluations."""

from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = 1
MAX_EXCERPT_CHARS = 6000
MAX_TRACEBACK_LINES = 28


def _tail(text: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _extract_status(error: str | None) -> str | None:
    if not error:
        return None
    match = re.match(r"^([A-Z_]+):", error.strip())
    if match:
        return match.group(1)
    return None


def _extract_exception(error: str | None) -> dict[str, str | None]:
    if not error:
        return {"type": None, "message": None}

    patterns = [
        r"(?P<type>[\w.]*DSLRuntimeError|[\w.]*RuntimeError|[\w.]*ValueError|"
        r"[\w.]*TypeError|[\w.]*AttributeError|[\w.]*AssertionError|"
        r"[\w.]*SyntaxError|[\w.]*Exception):\s*(?P<message>[^\n]+)",
        r"(?P<type>CUDA_ERROR_[A-Z_]+)\s*:?\s*(?P<message>[^\n]*)",
    ]
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, error))
    if not matches:
        return {"type": None, "message": None}

    match = matches[-1]
    return {
        "type": match.group("type"),
        "message": match.group("message").strip() or None,
    }


def _traceback_tail(error: str | None) -> list[str]:
    if not error:
        return []
    lines = [line.rstrip() for line in error.splitlines()]
    if not lines:
        return []
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            start = idx
    return lines[start:][-MAX_TRACEBACK_LINES:]


def _cute_compile_markers(error: str | None) -> dict[str, Any]:
    if not error:
        return {}
    preprocess = re.findall(r"Started preprocessing \[([^\]]+)\]", error)
    mangled = re.findall(r"Final mangled function name:\s*([^\n]+)", error)
    markers = {
        "preprocessed_functions": preprocess[-8:],
        "mangled_functions": [name.strip() for name in mangled[-6:]],
        "built_gpu_module": "Building GPU module" in error,
        "entered_gpu_module": "Entering GPU module" in error,
        "mentions_generate_mlir": "generate_mlir" in error,
        "mentions_ptxas": "ptxas" in error.lower(),
        "mentions_nvrtc": "nvrtc" in error.lower(),
    }
    return {key: value for key, value in markers.items() if value}


def _classify_level(
    *,
    compiled: bool,
    correct: bool,
    error: str | None,
    stats: dict[str, Any] | None,
) -> tuple[str, str]:
    status = _extract_status(error)
    err = error or ""
    low = err.lower()

    if correct and not error:
        return "passed", "benchmark_completed"
    if status == "STATIC_CUTE_SOURCE_ERROR":
        return "source_preflight", "host_source_static_scan"
    if status == "TIMEOUT" or "timeout" in low:
        return "timeout", "evaluation_timeout"
    if status in {
        "INCORRECT_SHAPE",
        "INCORRECT_NUMERICAL",
        "INCORRECT_DTYPE",
    }:
        return "correctness", "flashinfer_correctness_check"
    if status == "COMPILE_ERROR":
        return "cute_dsl_compiler", "solution_import_or_compile"
    if (
        "cute.compile" in err
        or "generate_mlir" in err
        or "dslruntimeerror" in low
        or "ptxas" in low
        or "nvrtc" in low
        or "mlir" in low
    ):
        return "cute_dsl_compiler", "cute_jit_mlir_or_codegen"
    if "module 'cutlass.cute' has no attribute" in low:
        return "cute_dsl_api", "cute_python_api_resolution"
    if any(
        token in low
        for token in (
            "illegal instruction",
            "illegal address",
            "misaligned address",
            "invalid configuration",
            "invalid device function",
            "cuda_error",
            "launch failed",
            "unspecified launch failure",
        )
    ):
        return "gpu_runtime", "cuda_kernel_launch_or_execution"
    if status == "RUNTIME_ERROR":
        return "host_runtime", "python_host_wrapper_or_runtime"
    if error:
        return "evaluator_error", "flashinfer_bench_evaluation"
    if stats and stats.get("trace_evaluations"):
        return "evaluator_error", "no_passed_trace"
    return "unknown", "unknown"


def _infer_likely_causes(level: str, error: str | None) -> list[str]:
    err = error or ""
    low = err.lower()
    causes: list[str] = []

    if "cute.load" in err or "cute.store" in err:
        causes.append(
            "Generated code used CUDA/Triton-style load/store helpers that are not "
            "available in this CuTe DSL package."
        )
    if "return is not allowed inside @cute.kernel" in err:
        causes.append(
            "Device code used an early return inside @cute.kernel; CuTe DSL needs "
            "predicated work instead."
        )
    if "int(...) inside @cute.kernel" in err or "cannot be interpreted as an integer" in err:
        causes.append(
            "A runtime DSL scalar was consumed as a Python int in kernel/JIT code; "
            "static loops and dynamic CuTe scalar control flow must stay separate."
        )
    if "expects argument" in err and ("got Int32" in err or "got Float32" in err):
        causes.append(
            "A decorated CuTe function argument is annotated or compiled as a Python "
            "host scalar, but the launcher passes a CuTe DSL scalar."
        )
    if "module 'cutlass.cute' has no attribute" in low:
        causes.append(
            "Generated code called a CuTe helper that is absent from the installed "
            "nvidia-cutlass-dsl runtime."
        )
    if "illegal instruction" in low:
        causes.append(
            "The GPU executed an unsupported or malformed instruction sequence; for "
            "WGMMA/TMEM/TMA this usually means the instruction shape, operand layout, "
            "alignment, or target architecture contract is wrong."
        )
    if "invalid configuration" in low:
        causes.append(
            "The launch configuration is invalid for the generated kernel, such as "
            "block size, cluster shape, dynamic shared memory, or resource usage."
        )
    if "misaligned" in low or "illegal address" in low:
        causes.append(
            "The kernel likely used an invalid address, stride, page index, or "
            "alignment for the generated memory instruction."
        )
    if level == "correctness":
        causes.append(
            "The kernel launched but did not match the reference shape, dtype, or "
            "numerics under FlashInfer Bench correctness checks."
        )

    if not causes:
        if level == "source_preflight":
            causes.append("The generated source failed before any B200 compile was attempted.")
        elif level == "cute_dsl_compiler":
            causes.append("CuTe DSL failed while lowering host-side generated Python into GPU IR.")
        elif level == "gpu_runtime":
            causes.append("The GPU kernel launched but failed during device execution.")
        elif level == "host_runtime":
            causes.append("The Python host wrapper failed before a valid benchmark result.")
        else:
            causes.append("The evaluator did not return enough structured information to isolate the cause.")
    return causes


def _infer_missing_to_fix(level: str, error: str | None) -> list[str]:
    missing: list[str] = []
    if level == "source_preflight":
        missing.append(
            "No MLIR, PTX, SASS, or NCU data exists for this proposal because the "
            "static source scan stopped before remote compilation."
        )
    if level in {"cute_dsl_compiler", "cute_dsl_api"}:
        missing.extend(
            [
                "Exact normalized source at the failing line and the decorated "
                "@cute.jit/@cute.kernel signature.",
                "Full CuTe DSL lowering log with preprocessing, mangled function "
                "name, generated MLIR stage, and any NVRTC/ptxas diagnostics.",
                "Generated PTX/SASS if lowering reached code generation; NCU cannot "
                "profile a kernel that never launches.",
            ]
        )
    if level == "gpu_runtime":
        missing.extend(
            [
                "NCU or NSYS report for the launched kernels, including launch "
                "parameters and device-side failure point.",
                "compute-sanitizer output for illegal address/misalignment cases.",
                "Generated PTX/SASS around the failing instruction to validate "
                "WGMMA/TMA/TMEM shape, layout, alignment, and architecture contracts.",
            ]
        )
    if level == "correctness":
        missing.extend(
            [
                "Reference-vs-output diffs for output and LSE, including max error "
                "locations.",
                "A dump of sparse_indices validity/masking cases for the failing trace.",
            ]
        )
    if level in {"host_runtime", "evaluator_error", "unknown"}:
        missing.append(
            "A complete FlashInfer Bench trace summary and full Python traceback from "
            "the host process."
        )
    return missing


def _why(level: str, phase: str, exception: dict[str, str | None]) -> str:
    message = exception.get("message")
    exc_type = exception.get("type")
    if message and exc_type:
        return f"{phase} failed with {exc_type}: {message}"
    if message:
        return f"{phase} failed: {message}"
    if level == "passed":
        return "The solution compiled, passed correctness, and produced benchmark timing."
    return f"{phase} failed; see evidence excerpts and raw error log."


def build_eval_diagnostics(
    *,
    compiled: bool,
    correct: bool,
    error: str | None,
    stats: dict[str, Any] | None = None,
    source_issues: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact structured diagnosis for one evaluation result."""
    level, phase = _classify_level(
        compiled=compiled, correct=correct, error=error, stats=stats
    )
    exception = _extract_exception(error)
    status = _extract_status(error)
    evidence = {
        "status": status,
        "compiled_flag": compiled,
        "correct_flag": correct,
        "exception": exception,
        "source_issues": source_issues or [],
        "traceback_tail": _traceback_tail(error),
        "cute_compile_markers": _cute_compile_markers(error),
        "log_excerpt_tail": _tail(error or ""),
        "trace_evaluations": (stats or {}).get("trace_evaluations", []),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome": "passed" if correct and compiled and not error else "failed",
        "level": level,
        "phase": phase,
        "host_device_boundary": (
            "Host CPU Python/CuTe DSL code compiles and launches work on the GPU; "
            "this level identifies where that host-to-device path failed."
        ),
        "why": _why(level, phase, exception),
        "likely_causes": _infer_likely_causes(level, error),
        "missing_to_fix": _infer_missing_to_fix(level, error),
        "evidence": evidence,
    }


def diagnostics_for_metric(metric: Any) -> dict[str, Any]:
    """Return existing diagnostics or build them from an EvalResult-like object."""
    diagnostics = getattr(metric, "diagnostics", None)
    if diagnostics:
        return diagnostics
    return build_eval_diagnostics(
        compiled=bool(getattr(metric, "compiled", False)),
        correct=bool(getattr(metric, "correct", False)),
        error=getattr(metric, "error", None),
        stats=getattr(metric, "stats", None),
    )

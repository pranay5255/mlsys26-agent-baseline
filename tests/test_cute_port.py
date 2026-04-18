from unittest.mock import patch

from agent.api import OPENROUTER_BASE_URL, create_inference_server
from agent.eval import CUTE_DSL_DEPENDENCIES, eval_kernel
from agent.main import build_parser
from agent.utils import (
    CUTE_DSL_SOURCE_SCAFFOLDING,
    find_cute_source_issues,
    load_config_from_yaml,
    normalize_cute_source,
)
from prompt.proposer_prompt import generate_proposer_prompt
from prompt.tuner_prompt import generate_tuner_prompt
from scripts.dsa_ps1_experiment_common import (
    ATTENTION_FUSED,
    build_agent_argv,
    build_experiment_focus,
    build_experiment_parser,
)

TASK_PARAMS = {
    "definition": '{"name": "toy", "inputs": []}',
    "target_gpu": "B200",
    "gpu_name": "B200",
    "gpu_architecture": "Blackwell",
    "dtype_str": "[]",
    "benchmark_warmup_runs": 3,
    "benchmark_iterations": 5,
    "benchmark_trials": 1,
    "eval_timeout_seconds": 180,
    "cute_dsl_version": "4.4.2",
}


def test_proposer_prompt_requires_cute_not_triton():
    prompt = generate_proposer_prompt(task_params=TASK_PARAMS)

    assert "CuTe DSL" in prompt
    assert "@cute.kernel" in prompt
    assert "@cute.jit" in prompt
    assert "cute.compile" in prompt
    assert "from cutlass.cute.runtime import from_dlpack" in prompt
    assert "Do not import from_dlpack from cutlass or cutlass.cute" in prompt
    assert "cutlass.cuda.default_stream" in prompt
    assert "cute.arch.thread_idx()" in prompt
    assert CUTE_DSL_SOURCE_SCAFFOLDING in prompt
    assert "bracket launch" in prompt
    assert "NO Triton imports" in prompt
    assert "triton.language as tl" not in prompt
    assert "Triton Version" not in prompt


def test_normalize_cute_source_fixes_common_generated_compile_errors():
    source = """
import torch
import cutlass.cute as cute
from cutlass.cute import from_dlpack
from cutlass import from_dlpack
from cutlass.cuda import default_stream

def run(x):
    stream = default_stream()
    stream.synchronize()
    return x
"""

    normalized = normalize_cute_source(source)

    assert "import cutlass" in normalized
    assert normalized.count("from cutlass.cute.runtime import from_dlpack") == 1
    assert "from cutlass.cute import from_dlpack" not in normalized
    assert "from cutlass import from_dlpack" not in normalized
    assert "from cutlass.cuda import default_stream" not in normalized
    assert "cutlass.cuda.default_stream()" in normalized
    assert "cutlass.cuda.stream_sync(stream)" in normalized


def test_normalize_cute_source_does_not_rewrite_correct_stream_helper():
    source = "import cutlass\nstream = cutlass.cuda.default_stream()\n"

    normalized = normalize_cute_source(source)

    assert "cutlass.cuda.cutlass.cuda.default_stream()" not in normalized
    assert normalized.count("cutlass.cuda.default_stream()") == 1


def test_normalize_cute_source_rewrites_triton_style_cute_builtins():
    source = """
import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    tid = cute.threadIdx.x
    bid = cute.blockIdx.y
    bdim = cute.blockDim.x
    cute.syncthreads()
"""

    normalized = normalize_cute_source(source)

    assert "cute.arch.thread_idx()[0]" in normalized
    assert "cute.arch.block_idx()[1]" in normalized
    assert "cute.arch.block_dim()[0]" in normalized
    assert "cute.arch.sync_threads()" in normalized


def test_find_cute_source_issues_flags_known_compile_dead_ends():
    source = """
import cutlass
import cutlass.cute as cute

@cute.jit
def launch_kernel(grid, block, smem_size, stream):
    x = cute.load(ptr, 0)
    cute.store(out, 0, x)
    y = cute.arch.shared_memory(0, cute.float32, (32,))
    kernel[grid, block, smem_size, stream](ptr, out)
"""

    issues = find_cute_source_issues(source)

    assert any("cute.load is not available" in issue for issue in issues)
    assert any("cute.store is not available" in issue for issue in issues)
    assert any("cute.arch.shared_memory is not available" in issue for issue in issues)
    assert any("bracket kernel launch is not valid" in issue for issue in issues)


def test_find_cute_source_issues_flags_kernel_runtime_traps():
    source = """
import cutlass.cute as cute

@cute.kernel
def kernel(x, n):
    if n > 0:
        return
    y = int(x)
"""

    issues = find_cute_source_issues(source)

    assert any("return is not allowed inside @cute.kernel" in issue for issue in issues)
    assert any("int(...) inside @cute.kernel" in issue for issue in issues)


def test_tuner_prompt_preserves_edit_protocol_and_cute_constraints():
    prompt = generate_tuner_prompt(
        previous_kernels=["def run(x):\n    return x"],
        previous_metrics=["correct=True speedup=1.0"],
        task_params=TASK_PARAMS,
    )

    assert "<old_str_1>" in prompt
    assert "<new_str_1>" in prompt
    assert "CuTe DSL Requirements" in prompt
    assert "Do not introduce Triton imports" in prompt
    assert "from cutlass.cute.runtime import from_dlpack" in prompt
    assert "cutlass.cuda.default_stream()" in prompt
    assert "cute.arch.thread_idx()" in prompt
    assert CUTE_DSL_SOURCE_SCAFFOLDING in prompt
    assert "bracket launch" in prompt
    assert "3 time(s)" in prompt
    assert "5 timed iteration(s)" in prompt


def test_experiment_focus_reaches_cli_and_prompts():
    parser = build_parser()
    args = parser.parse_args(["--experiment_focus", "prefer online softmax"])
    task_params = {**TASK_PARAMS, "experiment_focus": args.experiment_focus}

    proposer_prompt = generate_proposer_prompt(task_params=task_params)
    tuner_prompt = generate_tuner_prompt(
        previous_kernels=[],
        previous_metrics=[],
        task_params=task_params,
    )

    assert args.experiment_focus == "prefer online softmax"
    assert "Experiment Focus" in proposer_prompt
    assert "prefer online softmax" in proposer_prompt
    assert "Experiment Focus" in tuner_prompt
    assert "prefer online softmax" in tuner_prompt


def test_config_loader_respects_programmatic_argv(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("total_steps: 2\nexperiment_focus: from config\n")

    parser = build_parser()
    argv = [
        "--config",
        str(config_path),
        "--total_steps",
        "9",
        "--experiment_focus",
        "from argv",
    ]
    args = parser.parse_args(argv)
    args = load_config_from_yaml(args, parser, argv)

    assert args.total_steps == 9
    assert args.experiment_focus == "from argv"


def test_python_experiment_script_builds_modal_agent_argv():
    parser = build_experiment_parser(ATTENTION_FUSED)
    args = parser.parse_args(["--dry_run", "--total_steps", "1"])
    argv = build_agent_argv(ATTENTION_FUSED, args)

    assert "--eval_backend" in argv
    assert argv[argv.index("--eval_backend") + 1] == "modal"
    assert "--modal_gpu" in argv
    assert argv[argv.index("--modal_gpu") + 1] == "B200"
    assert "--experiment_focus" in argv
    assert "online or chunked reduction" in argv[argv.index("--experiment_focus") + 1]


def test_python_experiment_script_accepts_prompt_focus_append_and_override():
    parser = build_experiment_parser(ATTENTION_FUSED)
    args = parser.parse_args(
        [
            "--focus_append",
            "avoid early return",
            "--focus_append",
            "avoid arch shared memory",
        ]
    )
    focus = build_experiment_focus(ATTENTION_FUSED, args)

    assert "online or chunked reduction" in focus
    assert "avoid early return" in focus
    assert "avoid arch shared memory" in focus

    override_args = parser.parse_args(["--focus_override", "minimal compile-first"])
    override_focus = build_experiment_focus(ATTENTION_FUSED, override_args)

    assert override_focus == "minimal compile-first"


def test_openrouter_client_uses_openrouter_key_and_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("agent.api.openai.OpenAI") as openai_client:
        server = create_inference_server("openrouter")

    assert server.api_type == "openrouter"
    openai_client.assert_called_once()
    kwargs = openai_client.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert str(kwargs["base_url"]).rstrip("/") == OPENROUTER_BASE_URL


def test_eval_kernel_builds_python_cute_solution(monkeypatch):
    captured = {}

    class FakeTraceSet:
        def __init__(self):
            self.solutions = {}
            self._solution_by_name = {}

        @classmethod
        def from_path(cls, dataset_root):
            return cls()

    class FakeBenchmark:
        def __init__(self, trace_set, config):
            captured["solution"] = next(iter(trace_set._solution_by_name.values()))
            captured["config"] = config

        def run_all(self, dump_traces=False):
            class Result:
                traces = {"toy": []}

            return Result()

        def close(self):
            pass

    monkeypatch.setattr("agent.eval.TraceSet", FakeTraceSet)
    monkeypatch.setattr("agent.eval.Benchmark", FakeBenchmark)

    result = eval_kernel("def run():\n    return None", "toy", "/tmp/dataset")

    assert result.error == "No evaluation results"
    solution = captured["solution"]
    assert solution.spec.language.value == "python"
    assert solution.spec.entry_point == "main.py::run"
    assert solution.spec.dependencies == CUTE_DSL_DEPENDENCIES


def test_eval_kernel_rejects_static_cute_api_dead_end_before_benchmark(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("benchmark should not run for static CuTe source issues")

    monkeypatch.setattr("agent.eval.TraceSet.from_path", fail_if_called)

    result = eval_kernel(
        "import cutlass.cute as cute\n" "def run(x):\n" "    return cute.load(x, 0)\n",
        "toy",
        "/tmp/dataset",
    )

    assert result.compiled is False
    assert result.error.startswith("STATIC_CUTE_SOURCE_ERROR")

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smoke_dsa_definition_modal import run_smoke  # noqa: E402

from agent.main import main as agent_main  # noqa: E402


@dataclass(frozen=True)
class DsaPs1Experiment:
    name: str
    title: str
    agent_type: str
    total_steps: int
    pool_size: int
    temperature: float
    save_path: str
    focus: str


ATTENTION_FUSED = DsaPs1Experiment(
    name="attention_fused",
    title="Fused sparse attention",
    agent_type="evolve",
    total_steps=6,
    pool_size=5,
    temperature=0.45,
    save_path="outputs/experiments/dsa_ps1_attention_fused",
    focus=(
        "Generate a full fused sparse attention CuTe DSL solution for the DeepSeek "
        "DSA ps1 task. Treat each (token, query head) row as the core unit of "
        "work, use page_size=1 so sparse_indices directly address ckv/kpe cache "
        "rows, and avoid materializing a full [head, topk] logits tensor when an "
        "online or chunked reduction can compute logits, max, denominator, "
        "base-2 LSE, and value accumulation. Preserve exact masking of -1 "
        "indices, fp32 accumulation for logits/softmax/output sums, bf16 final "
        "output, and lse = logsumexp(logits * sm_scale) / log(2). Prefer one "
        "fused kernel if correct; otherwise use the smallest number of CuTe "
        "kernels needed for correctness and benchmark stability. Cache compiled "
        "executors by shape, dtype, stride/layout, device, and tuning parameters."
    ),
)


SOFTMAX_LSE = DsaPs1Experiment(
    name="softmax_lse",
    title="Softmax and LSE",
    agent_type="iterative",
    total_steps=6,
    pool_size=3,
    temperature=0.60,
    save_path="outputs/experiments/dsa_ps1_softmax_lse",
    focus=(
        "Generate a CuTe DSL solution for the same DSA ps1 task, with the main "
        "optimization effort on numerically stable softmax and base-2 LSE. Use a "
        "clear two-pass or online max/sumexp design over topk=2048 for each "
        "(token, head), explicitly mask sparse_indices == -1 before max and sum, "
        "and keep exponentials and reductions in fp32. It is acceptable to split "
        "logit generation, reduction/LSE, and output accumulation into multiple "
        "simple CuTe kernels if that makes correctness more reliable. Do not "
        "change the reference semantics: logits are q_nope dot ckv plus q_pe dot "
        "kpe, scaled by sm_scale; output is softmax(logits_scaled) @ ckv; lse is "
        "stored in fp32 base 2. Reuse module-level cute.compile caches during "
        "timed benchmark iterations."
    ),
)


RMSNORM_REDUCTION = DsaPs1Experiment(
    name="rmsnorm_reduction",
    title="RMSNorm-style row reduction",
    agent_type="evolve",
    total_steps=6,
    pool_size=5,
    temperature=0.75,
    save_path="outputs/experiments/dsa_ps1_rmsnorm_reduction",
    focus=(
        "Generate a CuTe DSL solution for the DSA ps1 task, but bias the design "
        "toward reusable row-reduction and normalization patterns similar to "
        "RMSNorm implementations. Do not implement RMSNorm or change attention "
        "math. Instead, structure the attention computation around stable "
        "per-row reductions for each (token, head): coalesced vector loads where "
        "possible, fp32 partial sums, deterministic reduction trees, reciprocal "
        "normalization of the softmax denominator, and coalesced bf16 output "
        "stores. Favor simple block/thread mappings that can later transfer to "
        "standalone RMSNorm-style kernels, while still exactly matching DSA "
        "sparse attention semantics and LSE base-2 output."
    ),
)


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def build_experiment_parser(experiment: DsaPs1Experiment) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run {experiment.title} on Modal B200"
    )
    parser.add_argument("--config", default="config/config_mini_test.yaml")
    parser.add_argument("--tasks_path", default="config/tasks_mini.txt")
    parser.add_argument("--save_path", default=experiment.save_path)
    parser.add_argument("--model_name", default="anthropic/claude-sonnet-4-5")
    parser.add_argument(
        "--total_steps",
        type=int,
        default=_env_int("TOTAL_STEPS", experiment.total_steps),
    )
    parser.add_argument("--pool_size", type=int, default=experiment.pool_size)
    parser.add_argument("--temperature", type=float, default=experiment.temperature)
    parser.add_argument(
        "--eval_timeout", type=int, default=_env_int("EVAL_TIMEOUT", 240)
    )
    parser.add_argument("--eval_warmup_runs", type=int, default=3)
    parser.add_argument("--eval_iterations", type=int, default=5)
    parser.add_argument("--eval_num_trials", type=int, default=1)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def build_agent_argv(
    experiment: DsaPs1Experiment, args: argparse.Namespace
) -> list[str]:
    argv = [
        "--config",
        args.config,
        "--tasks_path",
        args.tasks_path,
        "--test_source",
        "mlsys26-contest",
        "--eval_backend",
        "modal",
        "--modal_gpu",
        "B200",
        "--gpu_name",
        "B200",
        "--gpu_architecture",
        "Blackwell",
        "--api_type",
        "openrouter",
        "--model_name",
        args.model_name,
        "--agent_type",
        experiment.agent_type,
        "--total_steps",
        str(args.total_steps),
        "--pool_size",
        str(args.pool_size),
        "--temperature",
        str(args.temperature),
        "--eval_timeout",
        str(args.eval_timeout),
        "--eval_warmup_runs",
        str(args.eval_warmup_runs),
        "--eval_iterations",
        str(args.eval_iterations),
        "--eval_num_trials",
        str(args.eval_num_trials),
        "--max_completion_tokens",
        str(args.max_completion_tokens),
        "--save_path",
        args.save_path,
        "--experiment_focus",
        experiment.focus,
    ]
    if args.resume_from:
        argv.extend(["--resume_from", args.resume_from])
    if args.debug:
        argv.append("--debug")
    return argv


def run_experiment_script(
    experiment: DsaPs1Experiment, argv: Sequence[str] | None = None
) -> None:
    parser = build_experiment_parser(experiment)
    args = parser.parse_args(argv)
    agent_argv = build_agent_argv(experiment, args)

    print(f"Experiment: {experiment.title}")
    print(f"Save path: {args.save_path}")
    print("Agent command:")
    print("python -m agent.main " + " ".join(shlex.quote(a) for a in agent_argv))

    if args.dry_run:
        return

    _load_dotenv()
    if not args.smoke_only and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set. Export it or add it to .env.")

    if not args.skip_smoke:
        smoke_result = run_smoke()
        if not smoke_result.get("cute_empty_kernel_compiled"):
            raise RuntimeError(f"CuTe Modal smoke failed: {smoke_result}")

    if args.smoke_only:
        return

    agent_main(agent_argv)

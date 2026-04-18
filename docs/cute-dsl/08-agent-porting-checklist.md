# Agent Porting Checklist

## Prompts

- Proposer asks for complete Python/CuTe DSL code, not Triton.
- Tuner forbids Triton imports and preserves the `str_replace` edit protocol.
- Prompts include B200/Blackwell, CuTe DSL version, benchmark warmup/timed counts,
  timeout, dtype/input metadata, and the `run` entry point contract.
- Prompts should use `cutlass.cuda` stream helpers. In CuTe DSL 4.4.2,
  `cutlass.cute.cuda` is not an importable module.

## Evaluation

- FlashInfer Bench `BuildSpec.language` is `SupportedLanguages.PYTHON`.
- `entry_point` is `main.py::run`.
- `destination_passing_style` is `False`.
- Generated code is stored as `main.py` content.
- Eval timeout is high enough for CuTe JIT compilation on first run.

## Modal

- Modal image installs `nvidia-cutlass-dsl[cu13]`.
- GPU type defaults to `B200`.
- CuTe cache directory is writable.
- Debug mode enables CuTe line info/logging but normal mode keeps dumps off.
- Dataset volume sync is unchanged.

## OpenRouter

- `api_type: openrouter`.
- `OPENROUTER_API_KEY` is read from inherited env or `.env`.
- Base URL defaults to `https://openrouter.ai/api/v1`.
- Default model is `anthropic/claude-sonnet-4-5`.
- Do not log API key values.

## uv

- Dependencies live in `pyproject.toml`.
- `uv.lock` is committed.
- README install command is `uv sync`.
- `requirements.txt` is no longer the dependency source of truth.

## Smoke Commands

```bash
uv sync
uv run python -m compileall agent prompt
uv run python -c "import cutlass, cutlass.cute, modal, flashinfer_bench"
uv run python -m agent.main --config config/config_mini_test.yaml --eval_backend modal --modal_gpu B200
```

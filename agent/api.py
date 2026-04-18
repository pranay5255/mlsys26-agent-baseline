import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import openai

logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_TITLE = "mlsys26-cute-dsl-agent"


@dataclass(frozen=True)
class InferenceServer:
    """Small wrapper that keeps provider metadata with the SDK client."""

    api_type: str
    client: Any


def _load_dotenv() -> None:
    """Load .env values when available without overriding inherited env vars."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env(var_names: list[str], api_type: str) -> str:
    """Return the first configured env var value or raise a clear error."""
    for var_name in var_names:
        value = os.environ.get(var_name)
        if value:
            return value

    raise RuntimeError(
        f"Missing credentials for api_type='{api_type}'. Set one of: "
        + ", ".join(var_names)
    )


def create_inference_server(api_type: str):
    """Create an LLM client based on API type."""
    _load_dotenv()

    if api_type == "openai":
        return InferenceServer(
            api_type=api_type,
            client=openai.OpenAI(
                api_key=_require_env(["OPENAI_API_KEY"], api_type),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            ),
        )
    elif api_type == "openrouter":
        default_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_TITLE", OPENROUTER_DEFAULT_TITLE)
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        return InferenceServer(
            api_type=api_type,
            client=openai.OpenAI(
                api_key=_require_env(["OPENROUTER_API_KEY"], api_type),
                base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                default_headers=default_headers,
            ),
        )
    elif api_type in ("claude", "anthropic"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not api_key and not auth_token:
            _require_env(["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"], api_type)
        return InferenceServer(
            api_type=api_type,
            client=anthropic.Anthropic(api_key=api_key, auth_token=auth_token),
        )
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")


def _query_openai(client, model_name, prompt, max_completion_tokens, **kwargs):
    """Query OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_completion_tokens,
        **kwargs,
    )
    return response.choices[0].message.content


def _query_openrouter(client, model_name, prompt, max_completion_tokens, **kwargs):
    """Query OpenRouter through its OpenAI-compatible chat API."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        **kwargs,
    )
    return response.choices[0].message.content


def _query_anthropic(client, model_name, prompt, max_completion_tokens, **kwargs):
    """Query Anthropic API directly."""
    response = client.messages.create(
        model=model_name,
        max_tokens=max_completion_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return "".join(b.text for b in response.content if hasattr(b, "text"))


def query_inference_server(
    server,
    model_name: str,
    prompt: str,
    max_completion_tokens: int = 16384,
    retry_times: int = 5,
    **kwargs,
):
    """Query LLM with retry and exponential backoff."""
    kwargs.setdefault("temperature", 1.0)
    api_type = getattr(server, "api_type", None)
    client = getattr(server, "client", server)
    is_anthropic = api_type in ("claude", "anthropic") or isinstance(
        client, anthropic.Anthropic
    )
    if api_type == "openrouter":
        query_fn = _query_openrouter
    else:
        query_fn = _query_anthropic if is_anthropic else _query_openai

    for attempt in range(retry_times):
        try:
            return query_fn(client, model_name, prompt, max_completion_tokens, **kwargs)
        except Exception as e:
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{retry_times}): {e}"
            )
            if attempt == retry_times - 1:
                raise
            wait_time = (2**attempt) + random.uniform(0, 1)
            logger.info(f"Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)

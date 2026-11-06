"""OpenAI chat-completions client backed by the Quadrillion proxy.

The PostTrainBench judges call ``openai.OpenAI().chat.completions.create``. The
proxy serves OpenAI models only on the Responses API, and only streamed, with a
Quadrillion key. ``OpenAI`` here keeps the chat-completions call shape the
judges use and sends each call to the proxy's Responses API instead. It is the
only transport change from upstream; models, prompts and parsing are upstream's
apart from the judge model name.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai

#: Where the harness stages the box's proxy key (sandbox_llm_key).
KEY_FILES = (Path("/local/.sandbox-llm/env"), Path.home() / ".sandbox-llm/env")
DEFAULT_PROXY_URL = "https://api.quadrillion.io"
ATTRIBUTION = os.environ.get("QUADRILLION_ATTRIBUTION", "benchmark/ptb/judge")


def _staged() -> dict[str, str]:
    for path in KEY_FILES:
        if path.is_file():
            pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
            return {k.strip(): v.strip() for k, v in pairs}
    return {}


def _setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or _staged().get(name) or default
    if not value:
        raise RuntimeError(f"{name} is not set and no staged key file was found")
    return value


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


class _Completions:
    def __init__(self, client: openai.OpenAI):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **_ignored: Any,
    ) -> SimpleNamespace:
        system = "\n\n".join(_text(m["content"]) for m in messages if m["role"] == "system")
        params: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": m["role"], "content": _text(m["content"])}
                for m in messages
                if m["role"] != "system"
            ],
        }
        if system:
            params["instructions"] = system
        if max_completion_tokens or max_tokens:
            params["max_output_tokens"] = max_completion_tokens or max_tokens
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}
        with self._client.responses.stream(**params) as stream:
            response = stream.get_final_response()
        message = SimpleNamespace(role="assistant", content=response.output_text)
        return SimpleNamespace(
            model=model,
            choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
            usage=response.usage,
        )


class OpenAI:
    """Drop-in for ``openai.OpenAI`` as the PTB judges use it."""

    def __init__(self, *_args: Any, **_kwargs: Any):
        client = openai.OpenAI(
            base_url=_setting("QUADRILLION_PROXY_URL", DEFAULT_PROXY_URL),
            api_key=_setting("QUADRILLION_API_KEY"),
            default_headers={"X-Quadrillion-Attribution": ATTRIBUTION},
        )
        self.chat = SimpleNamespace(completions=_Completions(client))

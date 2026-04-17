"""LiteLLM-backed implementation of ``LLMProvider``.

Stage 6 widened the surface to match ``cytopert.providers.base``:
    * ``chat`` accepts ``api_base`` per-call so callers can route to
      different inference endpoints from the same provider instance,
      and we no longer mutate the module-global ``litellm.api_base``
      (which would race when several providers run concurrently).
    * Anthropic prompt caching is applied automatically by
      ``apply_anthropic_cache_control`` for ``claude*`` /
      ``anthropic/...`` model identifiers.
    * ``stream`` yields ``LLMResponseChunk`` objects via
      ``litellm.acompletion(stream=True)``.
    * ``count_tokens`` delegates to ``litellm.token_counter`` (falls
      back to a tokens-per-char heuristic if litellm cannot find a
      tokenizer for the chosen model).
    * ``LLMResponse.cost_usd`` is filled via
      ``litellm.completion_cost`` when available.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import litellm
from litellm import acompletion

from cytopert.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMResponseChunk,
    ToolCallRequest,
)
from cytopert.providers.prompt_caching import (
    apply_anthropic_cache_control,
    is_anthropic_model,
)

logger = logging.getLogger(__name__)

_KNOWN_PROVIDERS = {"openrouter", "deepseek", "anthropic", "openai", "vllm"}

# LiteLLM model prefixes per provider. Empty string = no prefix needed.
_MODEL_PREFIX = {
    "openrouter": "openrouter/",
    "deepseek": "deepseek/",
    "vllm": "hosted_vllm/",
    "openai": "openai/",
    "anthropic": "",
}

_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "vllm": "OPENAI_API_KEY",
}

# Suppress LiteLLM's "Provider List" startup banner once at import time.
litellm.suppress_debug_info = True


def _infer_provider(api_key: str | None, api_base: str | None, model: str) -> str:
    """Best-effort provider inference when callers don't pass provider_type."""
    base = (api_base or "").lower()
    mdl = (model or "").lower()
    if (api_key or "").startswith("sk-or-") or "openrouter" in base:
        return "openrouter"
    if "deepseek" in base or mdl.startswith("deepseek/") or "deepseek" in mdl:
        return "deepseek"
    if base:
        return "openai"
    if "anthropic" in mdl or "claude" in mdl:
        return "anthropic"
    if "gpt" in mdl or "openai" in mdl:
        return "openai"
    return "openai"


class LiteLLMProvider(LLMProvider):
    """LLM provider backed by LiteLLM (OpenRouter / DeepSeek / OpenAI / vLLM / Anthropic)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-sonnet-4-20250514",
        provider_type: str | None = None,
    ) -> None:
        super().__init__(api_key, api_base)
        self.default_model = default_model
        if provider_type and provider_type not in _KNOWN_PROVIDERS:
            raise ValueError(
                f"Unknown provider_type {provider_type!r}; expected one of {_KNOWN_PROVIDERS}"
            )
        self.provider_type = provider_type or _infer_provider(
            api_key, api_base, default_model
        )
        # Push the API key into the env so litellm can pick it up via the
        # vendor-specific env var. We do NOT mutate ``litellm.api_base``
        # here -- the legacy behaviour caused cross-instance races; per-
        # call kwargs in chat / stream supply the base URL instead.
        if api_key:
            env_var = _API_KEY_ENV[self.provider_type]
            os.environ.setdefault(env_var, api_key)
            if self.provider_type in {"openrouter", "openai", "vllm"}:
                os.environ[env_var] = api_key

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prefixed(self, model: str) -> str:
        prefix = _MODEL_PREFIX.get(self.provider_type, "")
        if not prefix or model.startswith(prefix):
            return model
        if "/" in model and self.provider_type == "openrouter":
            return f"openrouter/{model}"
        if "/" in model and self.provider_type != "openrouter":
            return model
        return f"{prefix}{model}"

    def _build_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int,
        temperature: float,
        api_base: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        prefixed = self._prefixed(model)
        # Apply Anthropic prefix-caching markers when the resolved model
        # belongs to the Claude family. Other providers ignore the
        # cache_control field but Anthropic returns 400 if we send it
        # against a non-Anthropic model behind LiteLLM, hence the gate.
        msgs = (
            apply_anthropic_cache_control(messages)
            if is_anthropic_model(prefixed)
            else messages
        )
        kwargs: dict[str, Any] = {
            "model": prefixed,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Per-call api_base override wins over the constructor default.
        chosen_base = api_base or self.api_base
        if chosen_base:
            kwargs["api_base"] = chosen_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
        return kwargs

    @staticmethod
    def _safe_completion_cost(response: Any) -> float | None:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception as exc:
            logger.debug("litellm.completion_cost unavailable: %s", exc)
            return None
        return float(cost) if cost is not None else None

    @staticmethod
    def _parse_usage(response: Any) -> dict[str, int]:
        usage_obj = getattr(response, "usage", None)
        if not usage_obj:
            return {}
        return {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }

    @staticmethod
    def _coerce_tool_calls(message: Any) -> list[ToolCallRequest]:
        calls: list[ToolCallRequest] = []
        raw = getattr(message, "tool_calls", None) if message is not None else None
        if not raw:
            return calls
        for tc in raw:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn else None
            args = getattr(fn, "arguments", None) if fn else None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            calls.append(
                ToolCallRequest(
                    id=getattr(tc, "id", "") or "",
                    name=name or "",
                    arguments=args or {},
                )
            )
        return calls

    # ------------------------------------------------------------------
    # LLMProvider implementation
    # ------------------------------------------------------------------

    def get_default_model(self) -> str:
        return self.default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=model or self.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_base=api_base,
            stream=False,
        )
        try:
            response = await acompletion(**kwargs)
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")
        choice = response.choices[0]
        message = choice.message
        return LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=self._coerce_tool_calls(message),
            finish_reason=getattr(choice, "finish_reason", None) or "stop",
            usage=self._parse_usage(response),
            cost_usd=self._safe_completion_cost(response),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
    ) -> AsyncIterator[LLMResponseChunk]:
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=model or self.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_base=api_base,
            stream=True,
        )
        response = await acompletion(**kwargs)
        async for chunk in response:
            choices = getattr(chunk, "choices", []) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None) or getattr(choice, "message", None)
            content_delta = getattr(delta, "content", None) if delta is not None else None
            tool_calls = self._coerce_tool_calls(delta) if delta is not None else []
            usage = self._parse_usage(chunk)
            yield LLMResponseChunk(
                content_delta=content_delta,
                tool_calls=tool_calls,
                finish_reason=getattr(choice, "finish_reason", None),
                usage=usage,
            )

    def count_tokens(
        self,
        text_or_messages: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        target = self._prefixed(model or self.default_model)
        try:
            if isinstance(text_or_messages, str):
                return int(
                    litellm.token_counter(model=target, text=text_or_messages)
                )
            return int(
                litellm.token_counter(model=target, messages=text_or_messages)
            )
        except Exception as exc:
            logger.debug("litellm.token_counter fallback (%s): %s", target, exc)
            # Crude tokens-per-character heuristic so callers always get a
            # finite number even when LiteLLM lacks a tokenizer for an
            # unusual model id (e.g. a vLLM-served local checkpoint).
            if isinstance(text_or_messages, str):
                return max(1, len(text_or_messages) // 4)
            text = " ".join(
                str(m.get("content", ""))
                for m in text_or_messages
                if isinstance(m, dict)
            )
            return max(1, len(text) // 4)

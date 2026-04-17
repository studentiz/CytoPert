"""LiteLLM provider implementation for CytoPert."""

import json
import os
from typing import Any

from litellm import acompletion

from cytopert.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_KNOWN_PROVIDERS = {"openrouter", "deepseek", "anthropic", "openai", "vllm"}

# LiteLLM model prefixes per provider. None = no prefix needed (e.g. anthropic
# / openai pick up the model name natively via env vars).
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


def _infer_provider(api_key: str | None, api_base: str | None, model: str) -> str:
    """Best-effort provider inference when caller didn't pass provider_type.

    Kept conservative because misclassification used to assign ``hosted_vllm/``
    to DeepSeek-via-OpenAI-compat (B1).
    """
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
    """LLM provider using LiteLLM (OpenRouter, Anthropic, OpenAI, vLLM, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-sonnet-4-20250514",
        provider_type: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        if provider_type and provider_type not in _KNOWN_PROVIDERS:
            raise ValueError(
                f"Unknown provider_type {provider_type!r}; expected one of {_KNOWN_PROVIDERS}"
            )
        self.provider_type = provider_type or _infer_provider(api_key, api_base, default_model)
        if api_key:
            os.environ.setdefault(_API_KEY_ENV[self.provider_type], api_key)
            if self.provider_type in {"openrouter", "openai", "vllm"}:
                os.environ[_API_KEY_ENV[self.provider_type]] = api_key
        if api_base:
            import litellm
            litellm.api_base = api_base
            litellm.suppress_debug_info = True

    def _prefixed(self, model: str) -> str:
        prefix = _MODEL_PREFIX.get(self.provider_type, "")
        if not prefix or model.startswith(prefix):
            return model
        if "/" in model and self.provider_type == "openrouter":
            return f"openrouter/{model}"
        if "/" in model and self.provider_type != "openrouter":
            return model
        return f"{prefix}{model}"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        model = self._prefixed(model or self.default_model)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            return LLMResponse(content=f"Error calling LLM: {str(e)}", finish_reason="error")

    def _parse_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args))
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    def get_default_model(self) -> str:
        return self.default_model

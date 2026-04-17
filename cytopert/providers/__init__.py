"""LLM providers for CytoPert (LiteLLMProvider lazily imported to avoid eager litellm load)."""

from typing import Any

from cytopert.providers.base import LLMProvider, LLMResponse, ToolCallRequest

__all__ = ["LLMProvider", "LLMResponse", "ToolCallRequest", "LiteLLMProvider"]


def __getattr__(name: str) -> Any:
    if name == "LiteLLMProvider":
        from cytopert.providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider
    raise AttributeError(f"module 'cytopert.providers' has no attribute {name!r}")

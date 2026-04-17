"""Base LLM provider interface.

Stage 6 of the completeness overhaul widened the abstraction:
    * ``LLMResponse`` carries a ``cost_usd`` field (LiteLLM can compute
      it for most providers via ``litellm.completion_cost``).
    * Concrete providers implement ``stream(...)`` (an async generator
      that yields ``LLMResponseChunk`` objects) and ``count_tokens(...)``
      so callers can do budget-aware compression without having to know
      provider-specific tokenizers.
    * ``chat`` formally accepts the ``api_base`` kwarg so providers can
      route per-call instead of relying on module-global state.

The ABC keeps default no-op implementations of the new methods so any
non-LiteLLM provider remains importable; callers that need streaming or
token counting must check ``hasattr(provider, ...)`` or rely on the
default raising ``NotImplementedError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider (single non-streamed completion)."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    #: Best-effort USD cost for this single call. ``None`` when the
    #: provider could not compute one (e.g. self-hosted vLLM).
    cost_usd: float | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class LLMResponseChunk:
    """Incremental piece of a streamed completion.

    ``content_delta`` carries the next text fragment; ``tool_calls``
    carries any tool_call updates emitted in this chunk (some providers
    spread a single call across multiple chunks). ``finish_reason`` is
    populated only on the terminal chunk.
    """

    content_delta: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request. Returns LLMResponse with content and/or tool_calls.

        ``api_base`` lets callers override the provider's default base
        URL on a per-call basis (useful when a single LiteLLM provider
        instance routes to multiple vLLM workers).
        """

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the default model identifier for this provider."""

    # ------------------------------------------------------------------
    # Optional capabilities. Default implementations raise so callers can
    # detect unsupported providers explicitly; concrete providers that
    # back streams / token counting override.
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
    ) -> AsyncIterator[LLMResponseChunk]:
        """Stream the next completion as ``LLMResponseChunk`` objects.

        Default raises ``NotImplementedError``. Concrete providers should
        override and yield chunks; callers must use ``async for`` /
        ``async with`` semantics.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming"
        )
        # The unreachable yield below is required so Python recognises
        # the method as an async generator even when overrides exist.
        if False:  # pragma: no cover
            yield LLMResponseChunk()

    def count_tokens(
        self,
        text_or_messages: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        """Return an integer token estimate for *text_or_messages*.

        Default raises ``NotImplementedError``. Concrete providers should
        delegate to a provider-specific tokenizer (e.g. tiktoken,
        Anthropic's token-counting endpoint, or LiteLLM's
        ``token_counter``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement count_tokens"
        )

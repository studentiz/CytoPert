"""Anthropic prompt caching (system_and_3 strategy).

Adapted from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:agent/prompt_caching.py
(MIT License). Near-verbatim port -- the helper is a small set of pure
functions that inject ``cache_control`` markers on up to 4 messages so
Anthropic prefix caching can save ~75% of input token cost on multi-turn
conversations. CytoPert's LiteLLMProvider calls this helper from
``chat`` whenever the active model name resolves to an Anthropic family
(``claude*`` / ``anthropic/...``). Other providers (DeepSeek, OpenAI,
Gemini, vLLM) skip it -- their caching is handled server-side or via
their own provider-specific knobs that LiteLLM forwards.

Differences from upstream:
    * Drop the ``cache_ttl="1h"`` upper-tier router -- CytoPert always
      uses the default 5-minute Anthropic ephemeral cache.
"""

from __future__ import annotations

import copy
from typing import Any


def _apply_cache_marker(
    msg: dict[str, Any],
    cache_marker: dict[str, Any],
    *,
    native_anthropic: bool = False,
) -> None:
    """Inject ``cache_control`` into a single message, handling content shapes.

    Three content shapes occur in practice:
        * ``str`` -- legacy chat messages. Promote to a list with a
          single text part so the cache marker has somewhere to live.
        * ``list[dict]`` -- already in the structured Anthropic /
          OpenAI tool-call format; mark the last part.
        * ``None`` / ``""`` -- assistant messages whose payload is
          tool_calls only; mark the message envelope directly.

    Tool messages are skipped on the OpenAI path because OpenAI's
    chat-completions schema does not allow ``cache_control`` on a tool
    role. On the native Anthropic path we mark the envelope.
    """
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: list[dict[str, Any]],
    *,
    native_anthropic: bool = False,
) -> list[dict[str, Any]]:
    """Apply the system_and_3 caching strategy and return a fresh list.

    Up to 4 ``cache_control`` breakpoints are placed: the system prompt
    plus the last 3 non-system messages. The input list is deep-copied
    so the caller's message buffer is not mutated.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker: dict[str, Any] = {"type": "ephemeral"}

    breakpoints_used = 0
    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [
        i for i in range(len(messages)) if messages[i].get("role") != "system"
    ]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def is_anthropic_model(model: str | None) -> bool:
    """Return True iff *model* names an Anthropic-family target.

    Recognises the LiteLLM prefix (``anthropic/...``) and the bare
    Anthropic SDK names (``claude-...``). Anything else is conservative
    and returns False so we never accidentally rewrite messages for
    providers that would 400 on the cache_control field.
    """
    if not model:
        return False
    low = model.lower()
    return low.startswith("anthropic/") or low.startswith("claude") or "/claude" in low

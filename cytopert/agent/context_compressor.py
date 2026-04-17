"""Default ContextEngine: protect head/tail, summarise the middle.

Adapted from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:agent/context_compressor.py
(MIT License). See docs/hermes-borrowing.md for the per-module diff
rationale.

Strategy (deliberately simpler than upstream):
    * Reserve ``protect_first_n`` system / user / assistant messages from
      the start of the conversation so the system prompt and the first
      few researcher framings always survive.
    * Reserve ``protect_last_n`` messages from the tail so the agent
      retains the most recent reasoning when answering the next turn.
    * Reserve every tool-result message that mentions a protected
      evidence id (``ContextEngine.is_protected``) so the chain-of-
      citations stays re-citable.
    * Everything in the middle that survives the previous filter is
      handed to a one-shot summariser LLM call. The summary replaces the
      compressed slice with a single ``role="system"`` message tagged
      ``[Compressed N earlier turns]``.
    * If anything goes wrong (no provider, summariser error, parsing
      failure), we fall back to returning the original message list
      unchanged. Losing context silently would be worse than letting
      the next call overflow the model's context window.

Differences from upstream:
    * No LCM (Latent Context Management) DAG; CytoPert ships only the
      one summarisation strategy.
    * No HEAD / TAIL token-budget allocation pass; the protect_first_n
      and protect_last_n knobs are absolute message counts, not token
      shares. This keeps the implementation under ~200 lines and
      matches the agentskills.io scope.
    * Adds explicit evidence-id protection via ``ContextEngine``.
"""

from __future__ import annotations

import logging
from typing import Any

from cytopert.agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

_FALLBACK_CONTEXT_LENGTH = 32_768


class CytoPertCompressor(ContextEngine):
    """Token-budget-aware compressor for CytoPert conversations.

    The summariser is invoked through the same provider object the
    AgentLoop uses; we cap its ``max_tokens`` at 1024 to avoid blowing
    the budget further in the act of compressing.
    """

    name = "compressor"

    def on_session_reset(self) -> None:
        """Clear per-session counters and the evidence-protect set.

        ContextEngine's default reset only clears token counters. The
        evidence-protect list is per-session by intent (a refuted chain
        in session A should not pin a tool message in session B), so we
        also reset it here. The CLI's ``/reset`` slash command calls
        this whenever the user clears history.
        """
        super().on_session_reset()
        self.evidence_id_protect = []

    def __init__(
        self,
        provider: Any | None = None,
        model: str | None = None,
        *,
        threshold_percent: float = 0.75,
        protect_first_n: int = 3,
        protect_last_n: int = 6,
        context_length: int = _FALLBACK_CONTEXT_LENGTH,
        summarise_max_tokens: int = 1024,
        summarise_temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self.threshold_percent = float(threshold_percent)
        self.protect_first_n = int(protect_first_n)
        self.protect_last_n = int(protect_last_n)
        self.context_length = int(context_length)
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.summarise_max_tokens = int(summarise_max_tokens)
        self.summarise_temperature = float(summarise_temperature)

    # ----- ContextEngine API -----------------------------------------

    def update_from_response(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(
            usage.get("total_tokens")
            or self.last_prompt_tokens + self.last_completion_tokens
        )

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        budget = self.threshold_tokens or int(
            self.context_length * self.threshold_percent
        )
        if budget <= 0:
            return False
        observed = (
            int(prompt_tokens)
            if prompt_tokens is not None
            else self.last_prompt_tokens
        )
        return observed >= budget

    def compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a compressed copy of *messages* (or the original on failure)."""
        if not messages:
            return messages
        n = len(messages)
        if n <= self.protect_first_n + self.protect_last_n:
            # Nothing to compress without violating the head/tail
            # protection contract.
            return messages

        head = messages[: self.protect_first_n]
        tail = messages[-self.protect_last_n :] if self.protect_last_n else []
        middle = messages[self.protect_first_n : n - self.protect_last_n]

        # Pull out evidence-protected messages from the middle and keep
        # them verbatim. They will sit just before the tail so the model
        # sees them in roughly chronological order.
        survivors: list[dict[str, Any]] = []
        compressible: list[dict[str, Any]] = []
        for msg in middle:
            if self.is_protected(msg):
                survivors.append(msg)
            else:
                compressible.append(msg)

        if not compressible:
            return messages

        summary = self._summarise(compressible)
        if summary is None:
            # Either we have no provider / model or the summariser raised.
            # Returning the original messages preserves correctness at the
            # cost of possibly hitting the model's context limit on the
            # next call -- preferable to silently dropping context.
            logger.warning(
                "context compression skipped: summariser unavailable or failed; "
                "returning %d messages unchanged",
                n,
            )
            return messages

        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": (
                f"[Compressed summary of {len(compressible)} earlier turns "
                f"(omitted by CytoPertCompressor to stay under "
                f"{self.threshold_tokens} prompt tokens)]\n\n{summary}"
            ),
        }
        self.compression_count += 1
        return [*head, summary_msg, *survivors, *tail]

    # ----- Internals -------------------------------------------------

    async def _summarise_async(
        self, messages: list[dict[str, Any]]
    ) -> str | None:
        if self.provider is None or not self.model:
            return None
        body = self._render_for_summary(messages)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a context compactor. Produce a dense, "
                    "information-preserving paragraph (no bullets) summarising "
                    "the conversation slice below. Keep tool names, evidence "
                    "ids, and chain ids verbatim; drop pleasantries; never "
                    "invent facts not present in the slice."
                ),
            },
            {"role": "user", "content": body},
        ]
        response = await self.provider.chat(
            messages=prompt,
            tools=None,
            model=self.model,
            max_tokens=self.summarise_max_tokens,
            temperature=self.summarise_temperature,
        )
        if response.finish_reason == "error" or not response.content:
            return None
        return str(response.content).strip() or None

    def _summarise(self, messages: list[dict[str, Any]]) -> str | None:
        """Synchronous wrapper used by ``compress``.

        ``compress`` is a sync method (the AgentLoop calls it from inside
        an ``async`` coroutine but expects a list back synchronously to
        keep the message list invariant local to the loop). If we are
        already in an event loop, run the coroutine via ``asyncio.run``
        in a worker thread; otherwise call it directly.
        """
        try:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop in this thread; safe to spin one up.
                return asyncio.run(self._summarise_async(messages))
            # Running loop already; spin up a fresh loop in a thread so we
            # do not deadlock the caller.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._summarise_async(messages))
                return future.result()
        except Exception as exc:  # noqa: BLE001 -- summariser is best-effort
            logger.warning("compressor summariser raised %s: %s", type(exc).__name__, exc)
            return None

    @staticmethod
    def _render_for_summary(messages: list[dict[str, Any]]) -> str:
        """Render the compressible slice into a stable plain-text form."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                )
            if content:
                parts.append(f"{role}: {content}")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                names = [
                    str(tc.get("function", {}).get("name", ""))
                    for tc in tool_calls
                    if isinstance(tc, dict)
                ]
                parts.append(f"{role}: [tool_calls -> {', '.join(filter(None, names))}]")
        return "\n".join(parts)

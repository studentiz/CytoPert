"""Pluggable context engine ABC.

Adapted from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:agent/context_engine.py (MIT License).
See docs/hermes-borrowing.md for the per-module diff rationale.

A context engine controls how the conversation context is managed when
the prompt grows close to the model's context-length budget. CytoPert
ships a single default implementation in
``cytopert.agent.context_compressor`` that summarises middle turns into
one synthetic system message; third-party engines can replace it later
through the plugin system in stage 7.3.

Differences from upstream:
    * Add ``evidence_id_protect``: a list of evidence ids that the
      compressor must NOT drop or summarise. CytoPert's downstream
      ``chains`` tool depends on those tool results being literally in
      the message history so the model can re-cite them.
    * No ``get_tool_schemas`` / ``handle_tool_call`` (CytoPert does not
      expose context-engine tools to the agent yet).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContextEngine(ABC):
    """Base class every CytoPert context engine must implement.

    Lifecycle:
        1. Engine is instantiated and assigned to ``AgentLoop.context_engine``.
        2. ``on_session_start(session_key)`` runs when ``process_direct``
           sees the session for the first time.
        3. ``update_from_response(usage)`` runs after every ``provider.chat``.
        4. ``should_compress()`` runs before each next ``provider.chat``.
        5. ``compress(messages)`` runs only when ``should_compress`` returned
           True; the returned list replaces the in-memory ``messages``.
        6. ``on_session_reset()`` runs when the user issues ``/reset`` in
           the interactive CLI.
    """

    # ----- Identity --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short engine identifier (e.g. ``"compressor"``)."""

    # ----- Token tracking (read by AgentLoop / CLI for display) -----

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # ----- Compaction parameters ------------------------------------

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # ----- CytoPert-specific extension -------------------------------

    #: Evidence ids whose tool result messages must remain literally in
    #: the rolling message list. Mutated by ``AgentLoop`` whenever a tool
    #: produces a new ``EvidenceEntry`` so the chain-of-citations stays
    #: re-citable even after compression. Order does not matter; the
    #: compressor treats it as a set.
    evidence_id_protect: list[str]

    def __init__(self) -> None:
        self.evidence_id_protect = []

    # ----- Core interface --------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: dict[str, Any]) -> None:
        """Refresh token state from a ``provider.chat`` usage dict."""

    @abstractmethod
    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        """Return True iff compaction must run before the next call."""

    @abstractmethod
    def compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a (possibly shorter) message list within budget."""

    # ----- Optional hooks --------------------------------------------

    def should_compress_preflight(self, messages: list[dict[str, Any]]) -> bool:
        """Cheap pre-flight check before the API call.

        Default returns False so the agent loop only relies on the
        post-call ``should_compress`` signal. Override when an engine has
        access to a quick token estimate.
        """
        return False

    def on_session_start(self, session_key: str, **kwargs: Any) -> None:
        """Called once per process when ``session_key`` is first seen."""

    def on_session_end(self, session_key: str, messages: list[dict[str, Any]]) -> None:
        """Called when a session truly ends (CLI exit, ``/reset``)."""

    def on_session_reset(self) -> None:
        """Reset per-session counters. Default clears token state."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    def update_model(self, model: str, context_length: int) -> None:
        """Update budget when the active model changes mid-process."""
        self.context_length = int(context_length)
        self.threshold_tokens = int(context_length * self.threshold_percent)

    # ----- Evidence protection helpers -------------------------------

    def protect_evidence(self, evidence_ids: list[str] | str) -> None:
        """Add evidence ids to the protect set."""
        ids = [evidence_ids] if isinstance(evidence_ids, str) else list(evidence_ids)
        for eid in ids:
            if eid and eid not in self.evidence_id_protect:
                self.evidence_id_protect.append(eid)

    def is_protected(self, message: dict[str, Any]) -> bool:
        """Return True iff *message* contains any protected evidence id.

        We scan the raw content text for the literal id; this avoids
        coupling to a specific JSON schema and works for any tool whose
        result mentions the evidence id verbatim (which CytoPert tools do
        by construction since the id is computed from the tool output).
        """
        if not self.evidence_id_protect:
            return False
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        else:
            text = str(content or "")
        return any(eid in text for eid in self.evidence_id_protect)

    # ----- Status ----------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Status dict for logging / display."""
        usage_pct = (
            min(100, self.last_prompt_tokens / self.context_length * 100)
            if self.context_length
            else 0
        )
        return {
            "engine": self.name,
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": usage_pct,
            "compression_count": self.compression_count,
            "protected_evidence_ids": list(self.evidence_id_protect),
        }

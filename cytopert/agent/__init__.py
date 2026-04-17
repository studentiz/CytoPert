"""Agent loop and context for CytoPert (lazy imports to avoid pulling LLM stack at import time)."""

from typing import Any

__all__ = ["AgentLoop", "ContextBuilder"]


def __getattr__(name: str) -> Any:
    if name == "AgentLoop":
        from cytopert.agent.loop import AgentLoop

        return AgentLoop
    if name == "ContextBuilder":
        from cytopert.agent.context import ContextBuilder

        return ContextBuilder
    raise AttributeError(f"module 'cytopert.agent' has no attribute {name!r}")

"""LLM-facing memory tool: add / replace / remove on three targets."""

from __future__ import annotations

import json
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.memory.store import MEMORY_TARGETS, MemoryStore, sanitize_entry


class MemoryTool(Tool):
    """Manage CytoPert persistent memory (context, researcher, hypothesis_log)."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return (
            "Persist information across sessions in one of three markdown stores: "
            "'context' (agent's environment/tool habits), 'researcher' (user profile/preferences), "
            "'hypothesis_log' (active mechanism chains). Actions: add | replace | remove. "
            "replace/remove use substring matching via old_text. Memory is frozen into the system "
            "prompt at session start, so changes take effect next session."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove"],
                    "description": "Mutation action.",
                },
                "target": {
                    "type": "string",
                    "enum": list(MEMORY_TARGETS),
                    "description": "Which memory store to modify.",
                },
                "content": {
                    "type": "string",
                    "description": "New content for add/replace.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Substring uniquely identifying the entry to replace/remove.",
                },
            },
            "required": ["action", "target"],
        }

    async def execute(
        self,
        action: str,
        target: str,
        content: str | None = None,
        old_text: str | None = None,
    ) -> str:
        try:
            if action == "add":
                result = self.store.add(target, sanitize_entry(content or ""))
            elif action == "replace":
                result = self.store.replace(target, old_text or "", sanitize_entry(content or ""))
            elif action == "remove":
                result = self.store.remove(target, old_text or "")
            else:
                return json.dumps(
                    {"success": False, "error": f"Unknown action {action!r}"},
                    ensure_ascii=False,
                )
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

"""Evidence tool: return or refresh structured evidence entries from current session / scverse outputs."""

from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.data.evidence_builder import build_evidence_summary
from cytopert.data.models import EvidenceEntry


class EvidenceTool(Tool):
    """
    Return or refresh structured evidence entries.
    Session/store can hold current evidence list; this tool returns summary for prompt.
    """

    def __init__(self, evidence_store: list[EvidenceEntry] | None = None) -> None:
        self._store: list[EvidenceEntry] = evidence_store if evidence_store is not None else []

    def set_store(self, store: list[EvidenceEntry]) -> None:
        """Set the evidence store (e.g. from workflow or other tools)."""
        self._store = store

    def add_entries(self, entries: list[EvidenceEntry]) -> None:
        """Append evidence entries."""
        self._store.extend(entries)

    @property
    def name(self) -> str:
        return "evidence"

    @property
    def description(self) -> str:
        return (
            "Get the current structured evidence summary (data and knowledge evidence entries). "
            "Use after running census or analysis tools to refresh evidence for mechanism reasoning."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_entries": {"type": "integer", "description": "Max number of entries to include (default 50)", "default": 50},
            },
            "required": [],
        }

    async def execute(self, max_entries: int = 50) -> str:
        summary = build_evidence_summary(self._store, max_entries=max_entries)
        return summary

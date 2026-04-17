"""Pathway tool: return pathway/topology constraint summary or validate mechanism (stub)."""

from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.knowledge.pathway import (
    check_mechanism_in_constraint,
    get_pathway_summary,
    get_topology_summary,
)


class PathwayConstraintTool(Tool):
    """Return pathway hierarchy and regulatory topology constraint summary for reasoning."""

    @property
    def name(self) -> str:
        return "pathway_constraint"

    @property
    def description(self) -> str:
        return "Get the current pathway hierarchy and regulatory network topology constraints. Use when reasoning about mechanisms."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self) -> str:
        return get_pathway_summary() + "\n\n" + get_topology_summary()


class PathwayCheckTool(Tool):
    """Check if a mechanism summary is consistent with pathway/topology constraints (stub)."""

    @property
    def name(self) -> str:
        return "pathway_check"

    @property
    def description(self) -> str:
        return "Check whether a proposed mechanism is consistent with pathway and topology constraints."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mechanism_summary": {"type": "string", "description": "Short summary of the mechanism to check"},
            },
            "required": ["mechanism_summary"],
        }

    async def execute(self, mechanism_summary: str) -> str:
        ok, msg = check_mechanism_in_constraint(mechanism_summary)
        return f"Constraint check: {'PASS' if ok else 'FAIL'}. {msg}"

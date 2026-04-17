"""chain_status tool: update MechanismChain lifecycle (proposed/supported/refuted/superseded)."""

from __future__ import annotations

import json
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.schema import CHAIN_STATUSES


class ChainStatusTool(Tool):
    """Transition a MechanismChain to a new status with evidence."""

    def __init__(self, store: ChainStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "chain_status"

    @property
    def description(self) -> str:
        return (
            "Transition a previously recorded MechanismChain to a new lifecycle status. "
            "Accepts: proposed | supported | refuted | superseded. Each status change is appended "
            "to the chain's JSONL audit trail. evidence_ids you supply are merged with existing ones."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chain_id": {"type": "string", "description": "Chain identifier returned by `chains`."},
                "status": {
                    "type": "string",
                    "enum": sorted(CHAIN_STATUSES),
                    "description": "New lifecycle status.",
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence IDs supporting this transition.",
                },
                "note": {"type": "string", "description": "Free-text note (e.g. wet-lab feedback)."},
            },
            "required": ["chain_id", "status"],
        }

    async def execute(
        self,
        chain_id: str,
        status: str,
        evidence_ids: list[str] | None = None,
        note: str = "",
    ) -> str:
        try:
            chain = self.store.update_status(
                chain_id=chain_id,
                status=status,
                evidence_ids=list(evidence_ids or []),
                note=note,
            )
        except (KeyError, ValueError) as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
        return json.dumps(
            {
                "success": True,
                "chain_id": chain.id,
                "status": status,
                "evidence_ids": chain.evidence_ids,
                "summary": chain.summary,
            },
            ensure_ascii=False,
        )

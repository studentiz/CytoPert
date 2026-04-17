"""chain_status tool: update MechanismChain lifecycle (proposed/supported/refuted/superseded)."""

from __future__ import annotations

import json
import logging
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.memory.store import MemoryStore
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.schema import CHAIN_STATUSES

logger = logging.getLogger(__name__)


class ChainStatusTool(Tool):
    """Transition a MechanismChain to a new status with evidence.

    A successful transition also appends a one-line entry to the
    ``hypothesis_log`` memory target so the latest status of every chain is
    visible across sessions in ``HYPOTHESIS_LOG.md`` (matching the README's
    "lifecycle is auditable everywhere" claim).
    """

    def __init__(self, store: ChainStore, memory: MemoryStore | None = None) -> None:
        self.store = store
        self.memory = memory

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

        memory_logged = False
        if self.memory is not None:
            # Keep the entry compact -- HYPOTHESIS_LOG has a 3000-char budget
            # and every chain transition writes exactly one line.
            entry = f"{chain.id} -> {status}: {chain.summary[:80]}"
            if note:
                entry += f" | {note[:80]}"
            try:
                result = self.memory.add("hypothesis_log", entry)
                memory_logged = bool(result.success)
                if not result.success:
                    # Hitting the memory budget is not a fatal error for the
                    # status transition itself -- record it in the tool
                    # response so the model can react (e.g. ask the user to
                    # consolidate hypothesis_log) but do not raise.
                    logger.warning(
                        "chain_status: HYPOTHESIS_LOG add rejected for %s: %s",
                        chain.id,
                        result.message,
                    )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "chain_status: HYPOTHESIS_LOG add raised for %s: %s", chain.id, exc
                )

        return json.dumps(
            {
                "success": True,
                "chain_id": chain.id,
                "status": status,
                "evidence_ids": chain.evidence_ids,
                "summary": chain.summary,
                "hypothesis_log_updated": memory_logged,
            },
            ensure_ascii=False,
        )

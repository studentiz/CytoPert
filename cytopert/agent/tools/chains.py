"""chains tool: submit / update mechanism chain candidate.

If a ChainStore is provided, the chain is persisted (status=proposed) so it can
be referenced cross-session and tracked through its lifecycle via `chain_status`.
"""

from __future__ import annotations

import json
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.data.models import MechanismChain, MechanismLink
from cytopert.persistence.chain_db import ChainStore


class ChainsTool(Tool):
    """Submit or update a MechanismChain candidate; optionally persist it."""

    def __init__(self, store: ChainStore | None = None) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "chains"

    @property
    def description(self) -> str:
        return (
            "Submit or update a mechanism chain candidate. Provide a short summary, links "
            "(from_node, to_node, relation), and evidence_ids. Returns suggested verification "
            "readout, priority (P1/P2), and the persisted chain_id you can later transition with "
            "`chain_status`."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Short summary of the mechanism chain"},
                "chain_id": {
                    "type": "string",
                    "description": "Optional id for updating an existing chain.",
                },
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_node": {"type": "string"},
                            "to_node": {"type": "string"},
                            "relation": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "description": "List of mechanism links",
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All evidence IDs cited",
                },
                "verification_readout": {
                    "type": "string",
                    "description": "Optional suggested experimental readout to test the chain.",
                },
            },
            "required": ["summary", "evidence_ids"],
        }

    async def execute(
        self,
        summary: str,
        evidence_ids: list[str],
        chain_id: str | None = None,
        links: list[dict[str, Any]] | None = None,
        verification_readout: str | None = None,
    ) -> str:
        link_list = links or []
        mechanism_links = []
        for link_dict in link_list:
            if not isinstance(link_dict, dict):
                continue
            mechanism_links.append(
                MechanismLink(
                    from_node=link_dict.get("from_node", ""),
                    to_node=link_dict.get("to_node", ""),
                    relation=link_dict.get("relation", ""),
                    evidence_ids=list(link_dict.get("evidence_ids", []) or []),
                )
            )
        priority = "P1" if len(evidence_ids) >= 2 else "P2"
        chain = MechanismChain(
            id=chain_id or "",
            links=mechanism_links,
            summary=summary,
            verification_readout=verification_readout
            or "Suggested: measure key node (gene/pathway) in perturbation vs control across states.",
            priority=priority,
            evidence_ids=list(evidence_ids),
        )
        persisted_id = chain.id
        if self.store is not None:
            try:
                persisted_id = self.store.upsert(chain, status="proposed", note="from chains tool")
            except Exception as e:
                return json.dumps(
                    {"success": False, "error": f"Failed to persist chain: {e}"},
                    ensure_ascii=False,
                )
        return json.dumps(
            {
                "success": True,
                "chain_id": persisted_id,
                "status": "proposed",
                "summary": chain.summary,
                "verification_readout": chain.verification_readout,
                "priority": chain.priority,
                "evidence_ids": chain.evidence_ids,
            },
            ensure_ascii=False,
        )

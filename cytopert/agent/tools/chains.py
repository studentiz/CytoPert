"""Chains tool: submit/update mechanism chain draft and request verification readouts."""

from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.data.models import MechanismChain, MechanismLink


class ChainsTool(Tool):
    """
    Submit or update a mechanism chain draft and request verification readout and priority.
    Output format: structured text or JSON for downstream workflow.
    """

    @property
    def name(self) -> str:
        return "chains"

    @property
    def description(self) -> str:
        return (
            "Submit or update a mechanism chain candidate. Provide a short summary of the chain, "
            "links (from_node, to_node, relation), and evidence_ids. Returns suggested verification readout and priority (P1/P2)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Short summary of the mechanism chain"},
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
                "evidence_ids": {"type": "array", "items": {"type": "string"}, "description": "All evidence IDs cited"},
            },
            "required": ["summary", "evidence_ids"],
        }

    async def execute(
        self,
        summary: str,
        evidence_ids: list[str],
        links: list[dict[str, Any]] | None = None,
    ) -> str:
        link_list = links or []
        mechanism_links = []
        for l in link_list:
            if not isinstance(l, dict):
                continue
            mechanism_links.append(MechanismLink(
                from_node=l.get("from_node", ""),
                to_node=l.get("to_node", ""),
                relation=l.get("relation", ""),
                evidence_ids=l.get("evidence_ids", []),
            ))
        chain = MechanismChain(
            id="chain_draft",
            links=mechanism_links,
            summary=summary,
            verification_readout="Suggested: measure key node (gene/pathway) in perturbation vs control across states.",
            priority="P1" if len(evidence_ids) >= 2 else "P2",
            evidence_ids=evidence_ids,
        )
        out = (
            f"Mechanism chain recorded.\nSummary: {chain.summary}\n"
            f"Verification readout: {chain.verification_readout}\nPriority: {chain.priority}\n"
            f"Evidence IDs: {chain.evidence_ids}"
        )
        return out

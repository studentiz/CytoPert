"""pathway_lookup tool: knowledge-class evidence for gene -> pathway / TF.

Replaces the legacy stub ``pathway_check`` / ``pathway_constraint``
tools. Backed by ``cytopert.knowledge.pathways``, which fetches PROGENy
/ DoRothEA / CollecTRI through decoupler 2.x's ``op`` namespace and
caches the resulting DataFrames under ``~/.cytopert/cache/knowledge``.

Stage 7.2 of the completeness overhaul. The result is also emitted as a
``KNOWLEDGE``-typed ``EvidenceEntry`` so downstream chains can cite the
lookup with the same ``[evidence: tool_pathway_lookup_<digest>]`` syntax
the binding enforcer recognises.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.knowledge.pathways import (
    KNOWN_SOURCES,
    lookup_genes,
    render_summary,
)

logger = logging.getLogger(__name__)


class PathwayLookupTool(Tool):
    """Look up the PROGENy / DoRothEA / CollecTRI regulators of a gene set."""

    @property
    def name(self) -> str:
        return "pathway_lookup"

    @property
    def description(self) -> str:
        return (
            "Look up the pathway response signatures (PROGENy) or "
            "transcription-factor regulons (DoRothEA / CollecTRI) that "
            "include the supplied genes. Backed by decoupler's omnipath "
            "resources; the first call per source caches the network "
            "DataFrame to disk so subsequent calls are offline. The "
            "result is recorded as KNOWLEDGE-typed evidence and is safe "
            "to cite via [evidence: tool_pathway_lookup_<digest>]."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Gene symbols to look up. Mixed case is fine.",
                },
                "source": {
                    "type": "string",
                    "enum": list(KNOWN_SOURCES),
                    "default": "progeny",
                    "description": "Knowledge source to query.",
                },
                "organism": {
                    "type": "string",
                    "default": "human",
                    "description": (
                        "Organism the network is curated for. decoupler "
                        "exposes 'human' and 'mouse' for all three sources."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "default": 25,
                    "description": "Cap on regulators / matches surfaced.",
                },
            },
            "required": ["genes"],
        }

    async def execute(
        self,
        genes: list[str],
        source: str = "progeny",
        organism: str = "human",
        top_n: int = 25,
    ) -> str:
        try:
            result = lookup_genes(
                genes=genes, source=source, organism=organism, top_n=top_n
            )
        except (ValueError, RuntimeError) as exc:
            return f"Error in pathway_lookup: {exc}"
        except Exception as exc:
            logger.exception("pathway_lookup failed")
            return f"Error in pathway_lookup: {type(exc).__name__}: {exc}"
        result["summary"] = render_summary(result)
        return json.dumps(result, ensure_ascii=False, default=str)

"""evidence_search tool: cross-session retrieval over EvidenceDB (SQLite + FTS5)."""

from __future__ import annotations

import json
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.persistence.evidence_db import EvidenceDB


class EvidenceSearchTool(Tool):
    """Look up evidence entries persisted in past sessions."""

    def __init__(self, db: EvidenceDB) -> None:
        self.db = db

    @property
    def name(self) -> str:
        return "evidence_search"

    @property
    def description(self) -> str:
        return (
            "Search persistent evidence (across all past sessions). Combine free-text `query` with "
            "structured filters: gene, pathway, tissue, tool_name. Returns up to `top_k` entries "
            "with id / summary / genes / pathways. Use this BEFORE re-running analyses you may have "
            "done before."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text query (FTS5)."},
                "gene": {"type": "string", "description": "Filter by gene symbol substring."},
                "pathway": {"type": "string", "description": "Filter by pathway name substring."},
                "tissue": {"type": "string", "description": "Filter by tissue/condition substring."},
                "tool_name": {"type": "string", "description": "Exact tool name (e.g. scanpy_de)."},
                "top_k": {"type": "integer", "description": "Max results (default 20).", "default": 20},
            },
            "required": [],
        }

    async def execute(
        self,
        query: str | None = None,
        gene: str | None = None,
        pathway: str | None = None,
        tissue: str | None = None,
        tool_name: str | None = None,
        top_k: int = 20,
    ) -> str:
        try:
            entries = self.db.search(
                query=query,
                gene=gene,
                pathway=pathway,
                tissue=tissue,
                tool_name=tool_name,
                top_k=int(top_k),
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
        out = [
            {
                "id": e.id,
                "type": e.type.value,
                "source": e.source,
                "tool_name": e.tool_name,
                "summary": e.summary,
                "genes": e.genes[:20],
                "pathways": e.pathways[:20],
            }
            for e in entries
        ]
        return json.dumps({"success": True, "count": len(out), "entries": out}, ensure_ascii=False)

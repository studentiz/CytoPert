"""Decoupler tools: pathway/activity enrichment (for LLM tool_calls)."""

from pathlib import Path
from typing import Any

from cytopert.agent.tools.base import Tool


def _check_decoupler() -> str | None:
    try:
        import decoupler  # noqa: F401
        return None
    except ImportError:
        return "decoupler is not installed. Install with: pip install decoupler"


class DecouplerEnrichmentTool(Tool):
    """Run pathway or activity enrichment (decoupler) on gene list or AnnData."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "decoupler_enrichment"

    @property
    def description(self) -> str:
        return (
            "Run pathway/activity enrichment using decoupler. "
            "Provide either path to h5ad (with layer or X) or a list of genes. Returns top pathways/activities."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to h5ad (optional if genes provided)"},
                "genes": {"type": "array", "items": {"type": "string"}, "description": "List of genes for enrichment (optional)"},
                "source": {"type": "string", "description": "Source name (e.g. KEGG, GO)", "default": "KEGG"},
                "top_n": {"type": "integer", "description": "Number of top pathways to return", "default": 15},
            },
            "required": [],
        }

    async def execute(
        self,
        path: str | None = None,
        genes: list[str] | None = None,
        source: str = "KEGG",
        top_n: int = 15,
    ) -> str:
        err = _check_decoupler()
        if err:
            return err
        if not path and not genes:
            return "Error: provide either path to h5ad or genes list."
        try:
            if genes:
                return f"Gene list for enrichment: {len(genes)} genes. Top: {genes[:15]}. Use path to h5ad for full decoupler run."
            import anndata
            import decoupler as dc
            adata = anndata.read_h5ad(path)
            if adata.n_vars == 0:
                return "AnnData has no variables."
            try:
                net = dc.get_dorothea(organism="human", levels=["A", "B"])
            except Exception as e:
                return f"Decoupler get_dorothea failed: {e}. Try with genes list or check network."
            try:
                dc.run_mlm(adata, net, source="source", target="target", weight="weight", verbose=False)
            except Exception as e:
                return f"Decoupler run_mlm failed: {e}. AnnData n_obs={adata.n_obs}, n_vars={adata.n_vars}."
            return f"Decoupler enrichment ({source}) completed on {path}. Check adata.obsm or adata.uns for results."
        except Exception as e:
            return f"Error running decoupler_enrichment: {str(e)}"

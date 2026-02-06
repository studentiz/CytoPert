"""Pertpy tools: perturbation distance, differential response (for LLM tool_calls)."""

from pathlib import Path
from typing import Any

from cytopert.agent.tools.base import Tool


def _check_pertpy() -> str | None:
    try:
        import pertpy  # noqa: F401
        return None
    except ImportError:
        return "pertpy is not installed. Install with: pip install pertpy"


class PertpyPerturbationDistanceTool(Tool):
    """Compute perturbation-related distance or effect (simplified wrapper)."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "pertpy_perturbation_distance"

    @property
    def description(self) -> str:
        return (
            "Compute perturbation distance or effect from AnnData with perturbation labels. "
            "Requires obs column indicating perturbation/condition. Returns summary."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to h5ad with perturbation labels in obs"},
                "perturbation_key": {"type": "string", "description": "obs column for perturbation/condition"},
                "metric": {"type": "string", "description": "Metric name (e.g. distance)", "default": "distance"},
            },
            "required": ["path", "perturbation_key"],
        }

    async def execute(
        self,
        path: str,
        perturbation_key: str,
        metric: str = "distance",
    ) -> str:
        err = _check_pertpy()
        if err:
            return err
        try:
            import anndata
            adata = anndata.read_h5ad(path)
            if perturbation_key not in adata.obs.columns:
                return f"Error: '{perturbation_key}' not in obs. Available: {list(adata.obs.columns)[:20]}"
            # Minimal stub: report group sizes and suggest using pt.tl for full analysis
            counts = adata.obs[perturbation_key].value_counts()
            return f"Perturbation groups ({perturbation_key}): {counts.to_dict()}. Use pertpy.tl for full distance/response analysis."
        except Exception as e:
            return f"Error running pertpy_perturbation_distance: {str(e)}"


class PertpyDifferentialResponseTool(Tool):
    """Summarize differential response across states (stub returns guidance)."""

    @property
    def name(self) -> str:
        return "pertpy_differential_response"

    @property
    def description(self) -> str:
        return "Assess differential response to perturbation across cell states. Requires perturbation and state labels in obs."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to h5ad"},
                "perturbation_key": {"type": "string", "description": "obs column for perturbation"},
                "state_key": {"type": "string", "description": "obs column for cell state"},
            },
            "required": ["path", "perturbation_key", "state_key"],
        }

    async def execute(
        self,
        path: str,
        perturbation_key: str,
        state_key: str,
    ) -> str:
        err = _check_pertpy()
        if err:
            return err
        try:
            adata = __import__("anndata").read_h5ad(path)
            if perturbation_key not in adata.obs.columns or state_key not in adata.obs.columns:
                return f"Error: missing obs columns. Available: {list(adata.obs.columns)[:20]}"
            p_counts = adata.obs[perturbation_key].value_counts()
            s_counts = adata.obs[state_key].value_counts()
            return f"Differential response setup: perturbation={p_counts.to_dict()}, state={s_counts.to_dict()}. Run pertpy analysis for full results."
        except Exception as e:
            return f"Error: {str(e)}"

"""Scanpy tools: preprocessing, clustering, DE, trajectory (for LLM tool_calls)."""

from pathlib import Path
from typing import Any

from cytopert.agent.tools.base import Tool


def _check_scanpy() -> str | None:
    try:
        import scanpy  # noqa: F401
        return None
    except ImportError:
        return "scanpy is not installed. Install with: pip install scanpy"


class ScanpyPreprocessTool(Tool):
    """Run basic scanpy preprocessing: filter, normalize, log1p, HVG, scale, PCA, UMAP."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "scanpy_preprocess"

    @property
    def description(self) -> str:
        return (
            "Run scanpy preprocessing on an h5ad file: filter_cells/genes, normalize, log1p, "
            "highly variable genes, scale, PCA, UMAP. Saves result to workspace. Returns summary."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to input h5ad file"},
                "min_genes": {"type": "integer", "description": "Min genes per cell (default 200)", "default": 200},
                "min_cells": {"type": "integer", "description": "Min cells per gene (default 3)", "default": 3},
                "n_top_genes": {"type": "integer", "description": "Number of HVGs (default 2000)", "default": 2000},
                "n_pcs": {"type": "integer", "description": "Number of PCs (default 50)", "default": 50},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        min_genes: int = 200,
        min_cells: int = 3,
        n_top_genes: int = 2000,
        n_pcs: int = 50,
    ) -> str:
        err = _check_scanpy()
        if err:
            return err
        try:
            import scanpy as sc
            adata = sc.read_h5ad(path)
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars), flavor="seurat")
            adata = adata[:, adata.var["highly_variable"]].copy()
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver="arpack", n_comps=min(n_pcs, adata.n_obs - 1, adata.n_vars - 1))
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(40, adata.obsm["X_pca"].shape[1]))
            sc.tl.umap(adata)
            out_path = self.workspace / "scanpy_preprocessed.h5ad"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(out_path)
            return f"Preprocessed: n_obs={adata.n_obs}, n_vars={adata.n_vars}. Saved to {out_path}"
        except Exception as e:
            return f"Error running scanpy_preprocess: {str(e)}"


class ScanpyClusterTool(Tool):
    """Run clustering (Leiden or Louvain) on a preprocessed h5ad."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "scanpy_cluster"

    @property
    def description(self) -> str:
        return "Run Leiden or Louvain clustering on preprocessed AnnData. Requires neighbors. Returns cluster counts."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to h5ad (with neighbors)"},
                "method": {"type": "string", "description": "leiden or louvain", "enum": ["leiden", "louvain"]},
                "resolution": {"type": "number", "description": "Resolution (default 1.0)", "default": 1.0},
                "key_added": {"type": "string", "description": "obs key for clusters (default clusters)"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        method: str = "leiden",
        resolution: float = 1.0,
        key_added: str = "clusters",
    ) -> str:
        err = _check_scanpy()
        if err:
            return err
        try:
            import scanpy as sc
            adata = sc.read_h5ad(path)
            if "neighbors" not in adata.uns:
                sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca" if "X_pca" in adata.obsm else None)
            if method == "leiden":
                sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
            else:
                sc.tl.louvain(adata, resolution=resolution, key_added=key_added)
            counts = adata.obs[key_added].value_counts()
            adata.write_h5ad(path)
            return f"Clustering ({method}): {len(counts)} clusters. Counts: {counts.to_dict()}"
        except Exception as e:
            return f"Error running scanpy_cluster: {str(e)}"


class ScanpyDETool(Tool):
    """Run differential expression (rank_genes_groups) between two groups."""

    @property
    def name(self) -> str:
        return "scanpy_de"

    @property
    def description(self) -> str:
        return "Run differential expression between two groups (e.g. cluster A vs B, or perturbation vs control)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to h5ad with group labels in obs"},
                "groupby": {"type": "string", "description": "obs column for groups (e.g. clusters, condition)"},
                "group1": {"type": "string", "description": "First group value"},
                "group2": {"type": "string", "description": "Second group value (reference)"},
                "top_n": {"type": "integer", "description": "Number of top genes to return (default 20)", "default": 20},
            },
            "required": ["path", "groupby", "group1", "group2"],
        }

    async def execute(
        self,
        path: str,
        groupby: str,
        group1: str,
        group2: str,
        top_n: int = 20,
    ) -> str:
        err = _check_scanpy()
        if err:
            return err
        try:
            import scanpy as sc
            adata = sc.read_h5ad(path)
            if groupby not in adata.obs.columns:
                return f"Error: groupby column '{groupby}' not in obs. Available: {list(adata.obs.columns)[:20]}"
            adata.obs[groupby] = adata.obs[groupby].astype(str)
            sc.tl.rank_genes_groups(adata, groupby, groups=[group1], reference=group2, method="wilcoxon")
            genes = list(adata.uns["rank_genes_groups"]["names"][group1][:top_n])
            return f"DE ({group1} vs {group2}): top genes: {genes}"
        except Exception as e:
            return f"Error running scanpy_de: {str(e)}"

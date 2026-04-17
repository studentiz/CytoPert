"""Census tool: query cellxgene_census or load local h5ad for AnnData/metadata."""

import asyncio
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.data.census_client import get_anndata, get_obs, load_local_h5ad


async def _run_with_timeout(fn, timeout_seconds: int, **kwargs: Any) -> Any:
    return await asyncio.wait_for(asyncio.to_thread(fn, **kwargs), timeout=timeout_seconds)


class CensusQueryTool(Tool):
    """Query cellxgene_census for AnnData slice (returns summary; full adata can be stored in session)."""

    @property
    def name(self) -> str:
        return "census_query"

    @property
    def description(self) -> str:
        return (
            "Query the cellxgene Census for single-cell data. Returns a summary of the AnnData slice "
            "(n_obs, n_vars, obs columns). Use obs_value_filter for cell filters (e.g. tissue), "
            "var_value_filter for gene filters. Optional: census_version."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "obs_value_filter": {"type": "string", "description": "SOMA filter for cells (e.g. tissue_ontology_term_id == 'UBERON:0000948')"},
                "var_value_filter": {"type": "string", "description": "SOMA filter for genes"},
                "census_version": {"type": "string", "description": "Census version (optional)"},
                "organism": {"type": "string", "description": "Organism (default: Homo sapiens)"},
                "timeout_seconds": {"type": "integer", "description": "Timeout for the Census query (seconds)", "default": 30},
                "obs_only": {"type": "boolean", "description": "Only fetch cell metadata (faster), ignores var_value_filter", "default": False},
                "obs_columns": {"type": "array", "items": {"type": "string"}, "description": "Optional obs column names to fetch"},
                "obs_coords": {"type": "string", "description": "Optional coords slice like 0:1000 (obs_only only)"},
                "max_cells": {"type": "integer", "description": "Max number of cells to fetch (default 20000)", "default": 20000},
            },
            "required": [],
        }

    async def execute(
        self,
        obs_value_filter: str | None = None,
        var_value_filter: str | None = None,
        census_version: str | None = None,
        organism: str = "Homo sapiens",
        timeout_seconds: int = 30,
        obs_only: bool = False,
        obs_columns: list[str] | None = None,
        obs_coords: str | None = None,
        max_cells: int = 20000,
    ) -> str:
        try:
            if obs_value_filter:
                obs_value_filter = (
                    obs_value_filter.replace(" AND ", " and ")
                    .replace(" OR ", " or ")
                    .replace("&&", " and ")
                    .replace("||", " or ")
                )
            if obs_only:
                coords = None
                if obs_coords:
                    if ":" in obs_coords:
                        start, end = obs_coords.split(":", 1)
                        coords = slice(int(start or 0), int(end) if end else None)
                    else:
                        coords = int(obs_coords)
                elif max_cells:
                    coords = slice(0, int(max_cells))
                params = {
                    "obs_value_filter": obs_value_filter or None,
                    "column_names": obs_columns,
                    "coords": coords,
                    "census_version": census_version,
                    "organism": organism,
                }
                obs = await _run_with_timeout(get_obs, timeout_seconds, **params)
                n_obs = getattr(obs, "shape", [len(obs)])[0]
                cols = list(getattr(obs, "columns", []))[:15]
                return (
                    f"Census obs query result: n_obs={n_obs}. "
                    f"obs columns: {cols}"
                )
            params = {
                "obs_value_filter": obs_value_filter or None,
                "var_value_filter": var_value_filter or None,
                "census_version": census_version,
                "organism": organism,
                "obs_coords": slice(0, int(max_cells)) if max_cells else None,
            }
            adata = await _run_with_timeout(get_anndata, timeout_seconds, **params)
            n_obs, n_vars = adata.n_obs, adata.n_vars
            obs_cols = list(adata.obs.columns)[:15]
            return (
                f"Census query result: n_obs={n_obs}, n_vars={n_vars}. "
                f"obs columns: {obs_cols}"
            )
        except asyncio.TimeoutError:
            return f"Error querying Census: timeout after {timeout_seconds}s (try narrower filters)"
        except Exception as e:
            return f"Error querying Census: {str(e)}"


class LoadLocalH5adTool(Tool):
    """Load a local h5ad file and return summary."""

    @property
    def name(self) -> str:
        return "load_local_h5ad"

    @property
    def description(self) -> str:
        return "Load a local h5ad file (path on disk). Returns summary: n_obs, n_vars, obs columns."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the h5ad file"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> str:
        try:
            adata = load_local_h5ad(path)
            n_obs, n_vars = adata.n_obs, adata.n_vars
            obs_cols = list(adata.obs.columns)[:15]
            return f"Loaded {path}: n_obs={n_obs}, n_vars={n_vars}. obs columns: {obs_cols}"
        except Exception as e:
            return f"Error loading h5ad: {str(e)}"

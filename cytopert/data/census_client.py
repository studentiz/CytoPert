"""Thin async-friendly wrappers around cellxgene-census APIs.

Provides ``open_census`` / ``get_anndata`` / ``get_obs`` / ``load_local_h5ad``.
The unused ``get_var`` wrapper was removed in stage 1; reintroduce it only
when a tool actually consumes per-gene metadata.
"""

from pathlib import Path
from typing import Any


def open_census(version: str | None = None):  # noqa: ANN201
    """Open the Census (SOMA). Returns context manager or census object per cellxgene_census API."""
    try:
        import cellxgene_census
        if version:
            return cellxgene_census.open_soma(census_version=version)
        return cellxgene_census.open_soma()
    except ImportError as e:
        raise ImportError("cellxgene_census is required. Install with: pip install cellxgene-census") from e


def get_anndata(
    obs_value_filter: str | None = None,
    var_value_filter: str | None = None,
    obs_column_names: list[str] | None = None,
    var_column_names: list[str] | None = None,
    obs_coords: Any | None = None,
    census_version: str | None = None,
    organism: str = "Homo sapiens",
):
    """
    Query Census and return an AnnData slice.
    obs_value_filter / var_value_filter are SOMA value filter expressions (e.g. "tissue_ontology_term_id == 'UBERON:0000948'").
    """
    try:
        import cellxgene_census
        kwargs: dict[str, Any] = {}
        if obs_value_filter:
            kwargs["obs_value_filter"] = obs_value_filter
        if var_value_filter:
            kwargs["var_value_filter"] = var_value_filter
        if obs_coords is not None:
            kwargs["obs_coords"] = obs_coords
        if obs_column_names:
            kwargs["obs_column_names"] = obs_column_names
        if var_column_names:
            kwargs["var_column_names"] = var_column_names
        if census_version:
            kwargs["census_version"] = census_version
        with open_census(census_version) as census:
            return cellxgene_census.get_anndata(census=census, organism=organism, **kwargs)
    except ImportError as e:
        raise ImportError("cellxgene_census is required. Install with: pip install cellxgene-census") from e


def get_obs(
    obs_value_filter: str | None = None,
    column_names: list[str] | None = None,
    coords: Any | None = None,
    census_version: str | None = None,
    organism: str = "Homo sapiens",
):
    """Get cell (obs) metadata from Census."""
    try:
        import cellxgene_census
        kwargs: dict[str, Any] = {}
        if obs_value_filter:
            kwargs["value_filter"] = obs_value_filter
        if column_names:
            kwargs["column_names"] = column_names
        if coords is not None:
            kwargs["coords"] = coords
        with open_census(census_version) as census:
            return cellxgene_census.get_obs(census=census, organism=organism, **kwargs)
    except ImportError as e:
        raise ImportError("cellxgene_census is required.") from e


def load_local_h5ad(path: str | Path):
    """Load a local h5ad file as AnnData."""
    try:
        import anndata
        return anndata.read_h5ad(path)
    except ImportError as e:
        raise ImportError("anndata is required. Install with: pip install anndata") from e

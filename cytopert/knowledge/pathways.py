"""Pathway / TF / regulon lookup backed by decoupler omnipath resources.

Replaces the legacy ``cytopert/knowledge/pathway.py`` stub. Three
sources are wired:
    * ``progeny``  -- 14 pathway response signatures (decoupler.op.progeny).
    * ``dorothea`` -- transcription-factor regulons curated by DoRothEA.
    * ``collectri`` -- the CollecTRI TF regulon resource.

Each source is fetched once per process via ``decoupler.op.<source>()``
and cached on disk under ``~/.cytopert/cache/`` so subsequent lookups
in the same workspace do not need network access. The helper
``lookup_genes`` returns the rows whose ``target`` column matches any
of the input genes (case-insensitive), shaped for downstream tools.

Stage 7.2 of the completeness overhaul. The ``pathway_lookup`` tool in
``cytopert/agent/tools/pathway_lookup.py`` wraps this module and turns
the result into ``EvidenceType.KNOWLEDGE`` evidence entries.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from cytopert.utils.helpers import ensure_dir, get_data_path

logger = logging.getLogger(__name__)

KNOWN_SOURCES = ("progeny", "dorothea", "collectri")

_CACHE_LOCK = threading.RLock()
_MEMORY_CACHE: dict[str, Any] = {}


def cache_dir() -> Path:
    return ensure_dir(get_data_path() / "cache" / "knowledge")


def _cache_key(source: str, organism: str) -> str:
    return f"{source}__{organism}"


def _cache_path(source: str, organism: str) -> Path:
    return cache_dir() / f"{_cache_key(source, organism)}.parquet"


def _fetch_from_decoupler(source: str, organism: str) -> Any:
    """Pull the network DataFrame from decoupler.op; raise if unavailable."""
    try:
        import decoupler as dc
    except ImportError as exc:
        raise RuntimeError(
            "decoupler is required for pathway_lookup; pip install decoupler"
        ) from exc
    op = getattr(dc, "op", None)
    if op is None:
        raise RuntimeError(
            "Installed decoupler is too old; pathway_lookup requires "
            "decoupler 2.x (which exposes dc.op.progeny / dorothea / collectri)."
        )
    fetcher = getattr(op, source, None)
    if fetcher is None:
        raise ValueError(
            f"Unknown decoupler resource {source!r}; expected one of {KNOWN_SOURCES}"
        )
    if source == "progeny":
        return fetcher(organism=organism, top=500)
    if source == "dorothea":
        return fetcher(organism=organism, levels=["A", "B", "C"])
    return fetcher(organism=organism)


def get_resource(source: str, organism: str = "human", *, force_refresh: bool = False) -> Any:
    """Return the network DataFrame for *source*, fetching + caching if needed.

    Successful fetches are written to ``~/.cytopert/cache/knowledge/<source>__<organism>.parquet``;
    subsequent calls in the same process hit the in-memory cache, and
    subsequent processes hit the on-disk parquet cache before falling
    back to the network.
    """
    if source not in KNOWN_SOURCES:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {KNOWN_SOURCES}"
        )
    key = _cache_key(source, organism)
    with _CACHE_LOCK:
        if not force_refresh and key in _MEMORY_CACHE:
            return _MEMORY_CACHE[key]
        path = _cache_path(source, organism)
        if not force_refresh and path.exists():
            try:
                import pandas as pd

                df = pd.read_parquet(path)
                _MEMORY_CACHE[key] = df
                return df
            except Exception as exc:
                logger.warning(
                    "Could not read cached %s; re-fetching. Reason: %s",
                    path,
                    exc,
                )
        df = _fetch_from_decoupler(source, organism)
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:
            logger.warning("Could not write cache %s: %s", path, exc)
        _MEMORY_CACHE[key] = df
        return df


def _normalise_genes(genes: list[str] | str | None) -> list[str]:
    if not genes:
        return []
    if isinstance(genes, str):
        items = [g.strip() for g in genes.replace(",", " ").split()]
    else:
        items = [str(g).strip() for g in genes]
    return [g for g in items if g]


def lookup_genes(
    genes: list[str] | str,
    source: str = "progeny",
    organism: str = "human",
    *,
    top_n: int = 25,
) -> dict[str, Any]:
    """Return a structured lookup result for *genes* in *source*.

    The returned dict has the shape:

    .. code-block:: python

        {
            "source": "progeny",
            "organism": "human",
            "genes_queried": ["NFATC1", "NOTCH1"],
            "matches": [
                {"gene": "NFATC1", "regulator": "NFkB", "weight": 0.4, ...},
                ...
            ],
            "regulators": ["NFkB", "JAK-STAT", ...],
            "n_rows_total": 1234,
            "n_matches": 17,
        }

    ``regulators`` lists the unique ``source`` (= pathway / TF) labels
    seen in the matched rows, in descending order of how many of the
    queried genes mentioned each label. This is what the
    ``pathway_lookup`` tool surfaces as the headline evidence summary.
    """
    queried = _normalise_genes(genes)
    if not queried:
        raise ValueError("lookup_genes: provide at least one gene symbol")
    df = get_resource(source, organism)
    # decoupler resources are always returned as a pandas DataFrame with
    # columns including 'source' (regulator/pathway) and 'target' (gene).
    # We lowercase both sides so case differences (NFATC1 vs Nfatc1) do
    # not silently miss.
    needles = {g.lower() for g in queried}
    target_lower = df["target"].astype(str).str.lower()
    mask = target_lower.isin(needles)
    matched = df[mask]
    matches: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for _, row in matched.head(top_n * 5).iterrows():
        regulator = str(row.get("source", ""))
        gene = str(row.get("target", ""))
        weight = row.get("weight")
        try:
            weight_val = float(weight) if weight is not None else None
        except (TypeError, ValueError):
            weight_val = None
        matches.append(
            {
                "gene": gene,
                "regulator": regulator,
                "weight": weight_val,
            }
        )
        counts[regulator] = counts.get(regulator, 0) + 1
    regulators = sorted(counts, key=lambda r: (-counts[r], r))[:top_n]
    return {
        "source": source,
        "organism": organism,
        "genes_queried": queried,
        "matches": matches[:top_n],
        "regulators": regulators,
        "n_rows_total": int(len(df)),
        "n_matches": int(matched.shape[0]),
    }


def render_summary(result: dict[str, Any]) -> str:
    """Render the lookup result for an LLM-facing tool reply."""
    if not result.get("genes_queried"):
        return "(no genes queried)"
    head = (
        f"pathway_lookup [{result['source']}/{result['organism']}] "
        f"matched {result['n_matches']} rows for "
        f"genes={result['genes_queried'][:8]}"
    )
    if len(result["genes_queried"]) > 8:
        head += " (+more)"
    if not result["regulators"]:
        return head + " -> no regulators / pathways found."
    regs = ", ".join(result["regulators"])
    return f"{head} -> regulators/pathways: {regs}"

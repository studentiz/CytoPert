"""Convert raw tool outputs into ``EvidenceEntry`` records.

The whitelist below is deliberately tight: only tools that produce
*scientifically reproducible* numerical evidence are persisted. Memory /
skills / chain / evidence_search outputs are not evidence sources -- they
are control-plane operations -- so they must not appear here. Stage 7.2
will add ``pathway_lookup`` (knowledge-class evidence) once the decoupler
backed implementation lands.
"""

import hashlib
import re
from typing import Any

from cytopert.data.models import EvidenceEntry, EvidenceType

_TOOLS_THAT_PRODUCE_EVIDENCE = {
    "census_query",
    "load_local_h5ad",
    "scanpy_preprocess",
    "scanpy_cluster",
    "scanpy_de",
}

_GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
_LIST_RE = re.compile(r"\[([^\[\]]+)\]")


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _extract_gene_candidates(text: str, limit: int = 25) -> list[str]:
    if not text:
        return []
    bracketed: list[str] = []
    for m in _LIST_RE.finditer(text):
        for token in m.group(1).split(","):
            t = token.strip().strip("'\"")
            if 2 <= len(t) <= 20 and re.fullmatch(r"[A-Za-z0-9_-]+", t):
                bracketed.append(t)
    if bracketed:
        seen: list[str] = []
        for g in bracketed:
            if g not in seen:
                seen.append(g)
            if len(seen) >= limit:
                break
        return seen
    return list(dict.fromkeys(_GENE_RE.findall(text)))[:limit]


def record_tool_evidence(
    tool_name: str,
    params: dict[str, Any],
    result: str,
    *,
    session_id: str = "",
    max_summary_chars: int = 600,
) -> EvidenceEntry | None:
    """Construct an EvidenceEntry from an arbitrary scverse/Census tool result string.

    Returns ``None`` for tools that should NOT generate evidence (memory, skills,
    chain_status, evidence_search, etc.) or when the result clearly indicates an error.
    """
    if not tool_name or tool_name not in _TOOLS_THAT_PRODUCE_EVIDENCE:
        return None
    text = (result or "").strip()
    if not text:
        return None
    if text.lower().startswith("error"):
        return None

    summary = text if len(text) <= max_summary_chars else text[: max_summary_chars - 3] + "..."
    digest = _short_hash(f"{tool_name}|{params}|{text}")
    eid = f"tool_{tool_name}_{digest}"
    genes = _extract_gene_candidates(text)
    pathways: list[str] = []
    state_conditions: list[str] = []
    for key in ("perturbation_key", "state_key", "groupby"):
        val = params.get(key) if isinstance(params, dict) else None
        if isinstance(val, str) and val:
            state_conditions.append(f"{key}={val}")
    source_bits: list[str] = []
    for key in ("path", "obs_value_filter", "var_value_filter", "census_version"):
        val = params.get(key) if isinstance(params, dict) else None
        if isinstance(val, str) and val:
            source_bits.append(f"{key}={val}")
    source = "; ".join(source_bits) or "tool_call"

    return EvidenceEntry(
        id=eid,
        type=EvidenceType.DATA,
        source=source,
        genes=genes,
        pathways=pathways,
        state_conditions=state_conditions,
        summary=summary,
        tool_name=tool_name,
        extra={"params": params if isinstance(params, dict) else {}},
    )


def build_evidence_summary(entries: list[EvidenceEntry], max_entries: int = 50) -> str:
    """Produce a text summary of evidence entries for injection into the system prompt."""
    if not entries:
        return "No evidence entries yet. Use census and analysis tools to generate evidence."
    lines = []
    for e in entries[-max_entries:]:
        lines.append(f"- [{e.id}] ({e.type.value}) {e.summary}")
        if e.genes:
            lines.append(f"  genes: {', '.join(e.genes[:15])}{'...' if len(e.genes) > 15 else ''}")
        if e.pathways:
            lines.append(f"  pathways: {', '.join(e.pathways[:10])}{'...' if len(e.pathways) > 10 else ''}")
    return "\n".join(lines)

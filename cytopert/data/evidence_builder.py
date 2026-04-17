"""Convert raw tool outputs into ``EvidenceEntry`` records.

The whitelist below is deliberately tight: only tools that produce
*scientifically reproducible* numerical evidence are persisted. Memory /
skills / chain / evidence_search outputs are not evidence sources -- they
are control-plane operations -- so they must not appear here. Stage 7.2
will add ``pathway_lookup`` (knowledge-class evidence) once the decoupler
backed implementation lands.

The gene extractor is a heuristic. It accepts mixed-case symbols (e.g. mouse
genes like ``Nfatc1``) but is intentionally pessimistic about short tokens
to keep noise low: short ``[bracketed]`` lists from tool output are trusted
verbatim, otherwise we fall back to a regex with a stop-word blacklist that
filters the most common English nouns / verbs / scverse field names that
look like gene symbols.
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
    "pathway_lookup",
}

#: Tool names whose results are stored as KNOWLEDGE-typed evidence
#: instead of DATA-typed evidence. Currently only the pathway lookup
#: emits KNOWLEDGE; other tools all return numerical reproducible
#: outputs and stay in the DATA bucket.
_TOOLS_PRODUCING_KNOWLEDGE = {"pathway_lookup"}

# Mixed-case alphanumeric tokens of length 2..10 starting with a letter,
# e.g. NFATC1 (human), Nfatc1 (mouse), WNT5a (mixed). The legacy
# uppercase-only pattern wrongly excluded mouse symbols entirely.
_GENE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]{1,9}\b")
_LIST_RE = re.compile(r"\[([^\[\]]+)\]")

# Stop list: common English / scverse tokens that look like gene symbols
# under the loose regex above. Compared case-insensitively.
_GENE_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "this", "that", "than", "then",
    "are", "was", "were", "has", "have", "had", "not", "but", "all", "any",
    "use", "via", "see", "set", "get", "got", "ran", "run", "log", "uns",
    "obs", "var", "dim", "raw", "key", "row", "col", "max", "min", "sum",
    "top", "len", "fdr", "lfc", "tpm", "cpm", "umi", "true", "false", "none",
    "null", "json", "yaml", "tissue", "celltype",
    "anndata", "scanpy", "pertpy", "decoupler", "leiden", "louvain", "umap",
    "pca", "preprocess", "cluster", "rank", "groupby", "group", "groups",
    "data", "type", "layer", "value", "filter", "result", "results", "summary",
    "method", "params", "p", "q",
})


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _looks_like_gene(token: str) -> bool:
    """Return True iff *token* could plausibly be a gene symbol."""
    if not (2 <= len(token) <= 10):
        return False
    if token.lower() in _GENE_STOPWORDS:
        return False
    # Pure numeric tokens are never gene symbols.
    if token.isdigit():
        return False
    # Must contain at least one letter.
    if not any(c.isalpha() for c in token):
        return False
    return True


def _extract_gene_candidates(text: str, limit: int = 25) -> list[str]:
    """Pull candidate gene symbols from a free-form tool result string.

    Bracketed lists win over the regex fallback because tool outputs that
    explicitly print ``[GENE1, GENE2, ...]`` are by convention real symbols,
    even if they fail the conservative ``_looks_like_gene`` check.
    """
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
    candidates = (t for t in _GENE_RE.findall(text) if _looks_like_gene(t))
    return list(dict.fromkeys(candidates))[:limit]


def record_tool_evidence(
    tool_name: str,
    params: dict[str, Any],
    result: str,
    *,
    session_id: str = "",
    max_summary_chars: int = 600,
) -> EvidenceEntry | None:
    """Construct an ``EvidenceEntry`` from an arbitrary scverse / Census tool result.

    Returns ``None`` for tools that should NOT produce evidence (memory /
    skills / chain_status / evidence_search etc.) or when the result clearly
    indicates an error so the agent loop can short-circuit cleanly.

    The ``session_id`` is preserved into ``extra`` so consumers (e.g.
    ``EvidenceDB.recent(session_id=...)``) can correlate entries with the
    session that produced them even after they've been re-loaded from disk.
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

    extra: dict[str, Any] = {"params": params if isinstance(params, dict) else {}}
    if session_id:
        extra["session_id"] = session_id

    evidence_type = (
        EvidenceType.KNOWLEDGE
        if tool_name in _TOOLS_PRODUCING_KNOWLEDGE
        else EvidenceType.DATA
    )
    return EvidenceEntry(
        id=eid,
        type=evidence_type,
        source=source,
        genes=genes,
        pathways=pathways,
        state_conditions=state_conditions,
        summary=summary,
        tool_name=tool_name,
        extra=extra,
    )


def build_evidence_summary(entries: list[EvidenceEntry], max_entries: int = 50) -> str:
    """Render evidence entries for injection into the system prompt.

    Mirrors the columns persisted in ``EvidenceDB`` so what the LLM sees is
    consistent with what ``evidence_search`` can later retrieve. Specifically
    we surface ``state_conditions`` and ``tool_name`` in addition to the
    summary / genes / pathways the legacy formatter already showed.
    """
    if not entries:
        return "No evidence entries yet. Use census and analysis tools to generate evidence."
    lines: list[str] = []
    for e in entries[-max_entries:]:
        head = f"- [{e.id}] ({e.type.value})"
        if e.tool_name:
            head += f" via {e.tool_name}"
        head += f": {e.summary}"
        lines.append(head)
        if e.genes:
            lines.append(
                f"  genes: {', '.join(e.genes[:15])}{'...' if len(e.genes) > 15 else ''}"
            )
        if e.pathways:
            lines.append(
                f"  pathways: {', '.join(e.pathways[:10])}{'...' if len(e.pathways) > 10 else ''}"
            )
        if e.state_conditions:
            lines.append(f"  state: {', '.join(e.state_conditions[:6])}")
    return "\n".join(lines)

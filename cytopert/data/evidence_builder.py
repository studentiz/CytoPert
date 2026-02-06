"""Build structured evidence entries from AnnData + perturbation/state + scverse tool outputs."""

from typing import Any

from cytopert.data.models import EvidenceEntry, EvidenceType


def from_anndata_obs(
    adata_obs_summary: dict[str, Any],
    dataset_id: str = "",
    tool_name: str | None = None,
) -> list[EvidenceEntry]:
    """
    Build evidence entries from AnnData obs summary (e.g. cell type counts, state labels).
    adata_obs_summary: dict with keys like 'cell_type_counts', 'state_groups', 'n_cells'.
    """
    entries = []
    eid = f"obs_{dataset_id or 'local'}_{hash(str(adata_obs_summary)) % 10**6}"
    summary_parts = []
    if "n_cells" in adata_obs_summary:
        summary_parts.append(f"n_cells={adata_obs_summary['n_cells']}")
    if "cell_type_counts" in adata_obs_summary:
        summary_parts.append(f"cell_types={list(adata_obs_summary.get('cell_type_counts', {}))}")
    if "state_groups" in adata_obs_summary:
        summary_parts.append(f"state_groups={adata_obs_summary.get('state_groups', [])}")
    entries.append(EvidenceEntry(
        id=eid,
        type=EvidenceType.DATA,
        source=dataset_id or "local",
        summary="; ".join(summary_parts) or "Obs metadata summary",
        tool_name=tool_name,
    ))
    return entries


def from_de_table(
    de_table: list[dict[str, Any]],
    group1: str = "",
    group2: str = "",
    dataset_id: str = "",
    tool_name: str = "scanpy_de",
    top_n: int = 20,
) -> list[EvidenceEntry]:
    """
    Build evidence entries from a differential expression table (e.g. scanpy rank_genes_groups output).
    de_table: list of dicts with keys like 'gene', 'logfoldchanges', 'pvals', 'group'.
    """
    entries = []
    genes = [r.get("gene", r.get("names", "")) for r in de_table[:top_n] if r]
    if not genes:
        return entries
    eid = f"de_{dataset_id or 'local'}_{group1}_{group2}_{hash(str(genes[:5])) % 10**6}"
    entries.append(EvidenceEntry(
        id=eid,
        type=EvidenceType.DATA,
        source=dataset_id or "local",
        genes=genes,
        summary=f"DE genes ({group1} vs {group2}): {', '.join(genes[:10])}{'...' if len(genes) > 10 else ''}",
        tool_name=tool_name,
    ))
    return entries


def from_perturbation_result(
    result_summary: dict[str, Any],
    perturbation_gene: str = "",
    dataset_id: str = "",
    tool_name: str = "pertpy",
) -> list[EvidenceEntry]:
    """
    Build evidence from perturbation analysis result (e.g. perturbation distance, differential response).
    result_summary: dict with keys like 'distance', 'affected_genes', 'state_dependent'.
    """
    entries = []
    eid = f"pert_{dataset_id or 'local'}_{perturbation_gene}_{hash(str(result_summary)) % 10**6}"
    summary_parts = [f"perturbation={perturbation_gene}"]
    if "distance" in result_summary:
        summary_parts.append(f"distance={result_summary.get('distance')}")
    if "affected_genes" in result_summary:
        summary_parts.append(f"affected_genes={result_summary.get('affected_genes', [])[:10]}")
    entries.append(EvidenceEntry(
        id=eid,
        type=EvidenceType.DATA,
        source=dataset_id or "local",
        genes=result_summary.get("affected_genes", [])[:20],
        summary="; ".join(summary_parts),
        tool_name=tool_name,
    ))
    return entries


def from_enrichment_result(
    pathways: list[str],
    genes: list[str],
    dataset_id: str = "",
    tool_name: str = "decoupler",
) -> list[EvidenceEntry]:
    """Build evidence from pathway/enrichment result."""
    entries = []
    eid = f"path_{dataset_id or 'local'}_{hash(str(pathways[:5])) % 10**6}"
    entries.append(EvidenceEntry(
        id=eid,
        type=EvidenceType.DATA,
        source=dataset_id or "local",
        pathways=pathways,
        genes=genes[:20],
        summary=f"Pathways: {', '.join(pathways[:10])}{'...' if len(pathways) > 10 else ''}",
        tool_name=tool_name,
    ))
    return entries


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

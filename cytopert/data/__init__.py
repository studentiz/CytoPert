"""Data access for CytoPert (Census wrappers, evidence model and builder).

Stage 1 of the completeness overhaul removed four factories that were never
called inside the package (``from_anndata_obs``, ``from_de_table``,
``from_perturbation_result``, ``from_enrichment_result``) and the unused
``get_var`` Census wrapper. Evidence creation now goes exclusively through
``record_tool_evidence`` from the agent loop.
"""

from cytopert.data.census_client import (
    get_anndata,
    get_obs,
    load_local_h5ad,
    open_census,
)
from cytopert.data.evidence_builder import (
    build_evidence_summary,
    record_tool_evidence,
)
from cytopert.data.models import EvidenceEntry, EvidenceType, MechanismChain, MechanismLink

__all__ = [
    "EvidenceEntry",
    "EvidenceType",
    "MechanismChain",
    "MechanismLink",
    "open_census",
    "get_anndata",
    "get_obs",
    "load_local_h5ad",
    "record_tool_evidence",
    "build_evidence_summary",
]

"""Pathway hierarchy and regulatory topology (stub)."""


def get_pathway_summary() -> str:
    """Return a short summary of pathway constraints for the prompt (stub)."""
    return "Pathway hierarchy: (stub) Use evidence-bound reasoning within known pathways. Replace with KEGG/Reactome or custom table."


def get_topology_summary() -> str:
    """Return regulatory network topology summary (stub)."""
    return "Regulatory topology: (stub) Reasoning must respect regulatory direction (TF -> target). Replace with GRN data."


def check_mechanism_in_constraint(mechanism_summary: str) -> tuple[bool, str]:
    """
    Check if a mechanism summary is consistent with pathway/topology constraints (stub).
    Returns (ok, message).
    """
    return True, "Stub: constraint check always passes. Add real pathway/topology data to enforce."

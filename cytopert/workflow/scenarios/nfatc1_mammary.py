"""Nfatc1 mammary development scenario: default config and run entry."""

from typing import Any

# Default config for Nfatc1 mammary (state/condition, perturbation gene, Census filters)
DEFAULT_CONFIG: dict[str, Any] = {
    "tissue_filter": ["mammary", "breast", "UBERON:0001911"],
    "perturbation_genes": ["Nfatc1"],
    "state_groups": ["basal", "luminal", "stem"],
    "census_obs_filter": None,  # e.g. "tissue_ontology_term_id == 'UBERON:0001911'"
    "local_h5ad_path": None,  # or path to local h5ad
}


def get_config() -> dict[str, Any]:
    """Return default scenario config (can be overridden by ~/.cytopert config)."""
    return dict(DEFAULT_CONFIG)


def run(
    research_question: str,
    scenario_config: dict[str, Any] | None = None,
    feedback: str | None = None,
) -> dict[str, Any]:
    """
    Run one round for Nfatc1 mammary scenario.
    feedback: optional experiment feedback for next round (e.g. new evidence or refuted chain).
    Returns dict with response text and optional structured chains.
    """
    from cytopert.config.loader import load_config
    from cytopert.workflow.pipeline import run_one_round, get_scenario_config
    import asyncio

    config = load_config()
    sc = scenario_config or get_config()
    if feedback:
        research_question += f"\n\nExperiment feedback from previous round: {feedback}"
    response = asyncio.run(run_one_round(config, research_question, data_config=sc, confirm_before_run=True))
    return {"response": response, "scenario": "nfatc1_mammary", "config": sc}

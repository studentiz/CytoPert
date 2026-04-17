"""Nfatc1 mammary development scenario.

Wraps the legacy single-round invocation as a one-stage ``Pipeline``
registered under ``nfatc1_mammary``. Behaviour matches the previous
``run`` entry point: build a prompt from the research question, the
default scenario config, and any ``--feedback`` payload, then hand it
to ``AgentLoop.process_direct`` via the standard agent turn stage.
"""

from __future__ import annotations

import asyncio
from typing import Any

from cytopert.config.loader import load_config
from cytopert.workflow.pipeline import (
    AgentTurnStage,
    Pipeline,
    StageContext,
    register_scenario,
    run_one_round,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "tissue_filter": ["mammary", "breast", "UBERON:0001911"],
    "perturbation_genes": ["Nfatc1"],
    "state_groups": ["basal", "luminal", "stem"],
    "census_obs_filter": None,  # e.g. "tissue_ontology_term_id == 'UBERON:0001911'"
    "local_h5ad_path": None,
}


def get_config() -> dict[str, Any]:
    """Return a copy of the default scenario config."""
    return dict(DEFAULT_CONFIG)


def _build_prompt(ctx: StageContext) -> str:
    prompt = ctx.research_question
    if ctx.data_config:
        prompt += f"\n\nData/perturbation config: {ctx.data_config}"
    if ctx.feedback:
        prompt += f"\n\nExperiment feedback from previous round: {ctx.feedback}"
    prompt += (
        "\n\n(Generate an execution plan first; after confirmation, run the tools.)"
    )
    return prompt


def build_pipeline() -> Pipeline:
    """Construct the Pipeline. Stage 7.1 ships a single agent turn."""
    return Pipeline(
        name="nfatc1_mammary",
        description=(
            "Mammary development perturbation: Nfatc1 KO across basal / luminal / "
            "stem populations."
        ),
        stages=[
            AgentTurnStage(
                name="plan_and_run",
                description=(
                    "Single-turn agent invocation: list a plan, then execute tools."
                ),
                prompt=_build_prompt,
            ),
        ],
    )


# Register at import time so cytopert.workflow.scenarios.__init__ picks
# us up via its autoimport pass.
register_scenario("nfatc1_mammary", build_pipeline)


def run(
    research_question: str,
    scenario_config: dict[str, Any] | None = None,
    feedback: str | None = None,
) -> dict[str, Any]:
    """Synchronous backward-compatible entry used by the legacy CLI path.

    Stage 7 keeps this function so callers that imported ``run`` directly
    still work, even though the CLI now resolves scenarios through
    ``SCENARIO_REGISTRY``.
    """
    config = load_config()
    sc = scenario_config or get_config()
    response = asyncio.run(
        run_one_round(
            config=config,
            research_question=research_question,
            data_config=sc,
            confirm_before_run=True,
            feedback=feedback,
        )
    )
    return {"response": response, "scenario": "nfatc1_mammary", "config": sc}

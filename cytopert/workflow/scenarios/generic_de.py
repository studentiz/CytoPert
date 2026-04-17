"""Generic single-cell DE workflow.

A domain-agnostic counterpart to the bundled ``nfatc1_mammary`` scenario:
takes any ``data_config`` payload describing the active dataset / column
names / contrast and asks the agent to plan and run a standard
preprocess -> DE -> pathway-lookup -> mechanism-chain pipeline.

Use this as the template when starting a new project. The CLI invokes
it as::

    cytopert run-workflow generic_de --question "..." \
        --feedback "<optional wet-lab feedback>"

The exact prompt the scenario sends is built from ``StageContext`` so
researchers can override individual fields via ``Config.workflow.generic_de``
in ``~/.cytopert/config.json``.
"""

from __future__ import annotations

from typing import Any

from cytopert.workflow.pipeline import (
    AgentTurnStage,
    Pipeline,
    StageContext,
    register_scenario,
)

DEFAULT_CONFIG: dict[str, Any] = {
    # Path to the input AnnData; if None, the agent is told to load via
    # ``census_query`` or to ask the researcher for an h5ad path.
    "h5ad_path": None,
    # Census filter expression to use when h5ad_path is None.
    "census_obs_filter": None,
    # Name of the obs column carrying the contrast (perturbation /
    # condition / treatment label).
    "contrast_column": "condition",
    # Group of interest in `contrast_column` (the "treated" arm).
    "treatment_group": None,
    # Reference group in `contrast_column` (the "control" arm).
    "control_group": None,
    # Optional state column (cell type / cluster / lineage).
    "state_column": None,
    # If a state column is provided, run the contrast within these
    # specific groups (left None = all groups).
    "state_groups": None,
    # Knowledge sources to query against the top DE genes.
    "pathway_sources": ["progeny", "dorothea"],
    # Organism the network resources are curated for.
    "organism": "human",
}


def get_config() -> dict[str, Any]:
    """Return a copy of the default scenario config."""
    return dict(DEFAULT_CONFIG)


def _build_prompt(ctx: StageContext) -> str:
    parts: list[str] = [ctx.research_question.strip() or
                        "Run the standard single-cell DE workflow on the dataset described below."]
    if ctx.data_config:
        parts.append("\n\n## Dataset / contrast configuration\n")
        for k, v in ctx.data_config.items():
            parts.append(f"- {k}: {v}")
    if ctx.feedback:
        parts.append(f"\n\n## Experiment feedback from previous round\n{ctx.feedback}")
    parts.append(
        "\n\n## Required steps\n"
        "1. Load the data (`load_local_h5ad` or `census_query`).\n"
        "2. Preprocess (`scanpy_preprocess`) and, if no state column is given, cluster (`scanpy_cluster`).\n"
        "3. Run `scanpy_de` for each state group, contrasting `treatment_group` vs `control_group`.\n"
        "4. For each top-N gene set, call `pathway_lookup` against the configured sources.\n"
        "5. Submit at least one mechanism chain via `chains` citing only real evidence ids.\n"
        "Generate a textual execution plan first; wait for the researcher to type "
        "`go` (or `execute` / `approve`) before invoking any tools."
    )
    return "".join(parts)


def build_pipeline() -> Pipeline:
    """Return the single-stage Pipeline registered as ``generic_de``."""
    return Pipeline(
        name="generic_de",
        description=(
            "Domain-agnostic single-cell DE template (any tissue / disease / "
            "perturbation). Wraps load + preprocess + DE + pathway lookup + "
            "chains in one agent turn."
        ),
        stages=[
            AgentTurnStage(
                name="plan_and_run",
                description="Single-turn agent invocation: plan, then execute tools.",
                prompt=_build_prompt,
            ),
        ],
    )


register_scenario("generic_de", build_pipeline)

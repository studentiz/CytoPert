"""Single-round pipeline: plan -> confirm -> compute -> evidence -> reasoning -> mechanism chains + verification."""

from typing import Any

from cytopert.agent.loop import AgentLoop
from cytopert.config.schema import Config
from cytopert.providers.litellm_provider import LiteLLMProvider


async def run_one_round(
    config: Config,
    research_question: str,
    data_config: dict[str, Any] | None = None,
    confirm_before_run: bool = True,
) -> str:
    """
    Run one round of the pipeline:
    research_question + data_config -> plan -> (optional confirm) -> agent with tools -> mechanism chains + verification.
    Returns final response (mechanism chains and verification readouts).
    """
    provider = LiteLLMProvider(
        api_key=config.get_api_key(),
        api_base=config.get_api_base(),
        default_model=config.agents.defaults.model,
        provider_type=config.get_provider_type(),
    )
    agent = AgentLoop(
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
    )
    # Build initial message with research question and optional data config
    prompt = research_question
    if data_config:
        prompt += f"\n\nData/perturbation config: {data_config}"
    if confirm_before_run:
        prompt += "\n\n(Generate an execution plan first; after confirmation, run the tools.)"
    response = await agent.process_direct(prompt, session_key="workflow:one_round")
    return response


def get_scenario_config(config: Config, scenario: str) -> dict[str, Any]:
    """Get workflow scenario config (e.g. nfatc1_mammary) from config.workflow."""
    return config.workflow.get(scenario, {})

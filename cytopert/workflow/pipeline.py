"""Workflow scaffolding for CytoPert.

A workflow scenario is a sequence of one or more ``Stage`` objects. The
agent is invoked once per stage with the prompt the stage produces;
each stage can read the previous stage's response from the
``StageContext`` so subsequent stages are able to refine the question
based on what the agent (and the underlying tool calls) have already
produced.

For the time being CytoPert ships a single bundled scenario
(``nfatc1_mammary``) wrapped as a ``Pipeline`` of one ``AgentTurnStage``
-- behaviour-wise identical to the legacy ``run_one_round`` so existing
callers keep working. The new abstraction is what stage 7.3 will plug
into when third-party packages contribute scenarios.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from cytopert.agent.loop import AgentLoop
from cytopert.config.schema import Config
from cytopert.providers.litellm_provider import LiteLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class StageContext:
    """Mutable state passed between successive ``Stage.run`` calls.

    Stage implementations may read previously-produced fields and add
    their own. ``responses[name]`` always carries the agent's final
    string for the named stage so later stages can reference it
    verbatim.
    """

    config: Config
    research_question: str
    data_config: dict[str, Any] = field(default_factory=dict)
    feedback: str | None = None
    session_key: str = "workflow:one_round"
    responses: dict[str, str] = field(default_factory=dict)


@dataclass
class StageResult:
    """Outcome of a single stage."""

    name: str
    response: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage:
    """A single conversational step inside a Pipeline.

    ``run`` receives the shared ``StageContext`` plus the active
    ``AgentLoop`` and returns a ``StageResult``. The default ``run`` is
    set to ``None`` so subclasses (or ``AgentTurnStage``) can override
    cleanly.
    """

    name: str
    description: str = ""
    run: Callable[["AgentLoop", StageContext], Awaitable[StageResult]] | None = None


@dataclass
class Pipeline:
    """An ordered list of stages bound to a scenario name."""

    name: str
    stages: list[Stage]
    description: str = ""

    async def run(self, agent: AgentLoop, ctx: StageContext) -> dict[str, Any]:
        """Execute every stage in order; return the final-stage payload."""
        if not self.stages:
            raise ValueError(f"Pipeline {self.name!r} has no stages")
        last: StageResult | None = None
        for stage in self.stages:
            if stage.run is None:
                raise TypeError(f"Stage {stage.name!r} has no run callable")
            last = await stage.run(agent, ctx)
            ctx.responses[stage.name] = last.response
        assert last is not None
        return {
            "scenario": self.name,
            "response": last.response,
            "stage": last.name,
            "extra": last.extra,
            "responses": dict(ctx.responses),
        }


def _agent_turn_runner(prompt_builder: Callable[[StageContext], str]):
    async def _run(agent: AgentLoop, ctx: StageContext) -> StageResult:
        prompt = prompt_builder(ctx)
        response = await agent.process_direct(
            prompt, session_key=ctx.session_key, user_feedback=ctx.feedback
        )
        return StageResult(name="(unnamed)", response=response)
    return _run


def agent_turn_stage(
    *,
    name: str,
    description: str = "",
    prompt: Callable[[StageContext], str],
) -> Stage:
    """Convenience constructor for a single agent-turn stage.

    ``prompt`` builds the user prompt from the shared ``StageContext``
    so a stage can mix in research_question / data_config / feedback /
    earlier-stage responses without having to subclass ``Stage``.
    """
    inner = _agent_turn_runner(prompt)

    async def _named(agent: AgentLoop, ctx: StageContext) -> StageResult:
        result = await inner(agent, ctx)
        result.name = name
        return result

    return Stage(name=name, description=description, run=_named)


# Backwards-compatible alias (the original API used CamelCase because the
# helper acts like a class constructor; we keep the old name reachable so
# external scenarios written against the alpha do not break).
AgentTurnStage = agent_turn_stage


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY: dict[str, Callable[[], Pipeline]] = {}


def register_scenario(name: str, factory: Callable[[], Pipeline]) -> None:
    """Register a scenario factory.

    Re-registration with the same name overrides the previous factory
    and emits a warning so downstream callers (the CLI's run_workflow
    handler) always see the most recent definition. The cytopert
    plugins manager calls this from plugin ``setup`` hooks; built-in
    scenarios call it from their module top-level (loaded by
    ``cytopert.workflow.scenarios``'s autoimport).
    """
    if name in SCENARIO_REGISTRY:
        logger.warning(
            "Scenario %r re-registered; previous factory will be replaced", name
        )
    SCENARIO_REGISTRY[name] = factory


def get_scenario(name: str) -> Pipeline | None:
    """Return a fresh Pipeline for *name* (or None if not registered)."""
    factory = SCENARIO_REGISTRY.get(name)
    return factory() if factory else None


def available_scenarios() -> list[str]:
    """Return registered scenario names sorted alphabetically."""
    # Touch the scenarios package once so its autoimport runs lazily
    # before anyone enumerates the registry.
    try:
        import cytopert.workflow.scenarios  # noqa: F401
    except Exception as exc:
        logger.debug("scenarios autoimport failed: %s", exc)
    return sorted(SCENARIO_REGISTRY)


# ---------------------------------------------------------------------------
# Legacy single-round entry (still used by CLI and tests)
# ---------------------------------------------------------------------------


async def run_one_round(
    config: Config,
    research_question: str,
    data_config: dict[str, Any] | None = None,
    confirm_before_run: bool = True,
    *,
    session_key: str = "workflow:one_round",
    feedback: str | None = None,
) -> str:
    """Run one agent turn against the active ``Config``.

    This is the legacy single-stage path: build a provider + AgentLoop,
    construct a one-shot prompt and return the agent's final reply.
    Kept stable for the CLI's existing nfatc1_mammary scenario.
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
        max_tokens=config.agents.defaults.max_tokens,
        temperature=config.agents.defaults.temperature,
    )
    prompt = research_question
    if data_config:
        prompt += f"\n\nData/perturbation config: {data_config}"
    if confirm_before_run:
        prompt += "\n\n(Generate an execution plan first; after confirmation, run the tools.)"
    return await agent.process_direct(
        prompt, session_key=session_key, user_feedback=feedback
    )


def get_scenario_config(config: Config, scenario: str) -> dict[str, Any]:
    """Return user-supplied scenario config from ``Config.workflow``."""
    return config.workflow.get(scenario, {})

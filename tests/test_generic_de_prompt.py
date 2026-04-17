"""Regression tests for cytopert.workflow.scenarios.generic_de._build_prompt.

Pins the multi-line bullet shape of the rendered Dataset / contrast
configuration block. The historical bug appended each ``- key: value``
without a trailing newline; under ``"".join(parts)`` successive bullets
collapsed onto a single line (``- a: 1- b: 2``) and the LLM read the
config as one mangled bullet.
"""

from __future__ import annotations

from cytopert.config.loader import load_config
from cytopert.workflow.pipeline import StageContext
from cytopert.workflow.scenarios.generic_de import (
    DEFAULT_CONFIG,
    _build_prompt,
)


def _ctx(**overrides) -> StageContext:
    """Build a StageContext with sensible defaults for a prompt-only test."""
    base = dict(
        config=load_config(),
        research_question="Run a generic DE workflow.",
        data_config=dict(DEFAULT_CONFIG),
        feedback=None,
        session_key="workflow:test",
    )
    base.update(overrides)
    return StageContext(**base)


def test_dataset_config_bullets_each_on_their_own_line() -> None:
    cfg = {"contrast_column": "condition", "treatment_group": "pert", "control_group": "ctrl"}
    out = _build_prompt(_ctx(data_config=cfg))
    # Header + one bullet line per key, in insertion order. The historical
    # bug rendered "- a: 1- b: 2" on a single line; assert per-line shape.
    assert "## Dataset / contrast configuration" in out
    assert "- contrast_column: condition\n- treatment_group: pert\n- control_group: ctrl\n" in out
    # Defensive: never see two bullets glued together.
    assert "- " in out and "1- " not in out and "ctrl- " not in out


def test_no_data_config_skips_the_section() -> None:
    out = _build_prompt(_ctx(data_config={}))
    assert "## Dataset / contrast configuration" not in out
    # Required-steps block is always present.
    assert "## Required steps" in out


def test_feedback_appended_before_required_steps() -> None:
    out = _build_prompt(_ctx(feedback="qPCR refuted chain_0001"))
    fb_idx = out.index("## Experiment feedback from previous round")
    steps_idx = out.index("## Required steps")
    assert fb_idx < steps_idx
    assert "qPCR refuted chain_0001" in out


def test_default_research_question_when_blank() -> None:
    out = _build_prompt(_ctx(research_question="   "))
    assert out.startswith(
        "Run the standard single-cell DE workflow on the dataset described below."
    )

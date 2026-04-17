"""Stage 2.2 tests for the evidence-gate research-conclusion classifier."""

from __future__ import annotations

import pytest

from cytopert.agent.loop import AgentLoop


@pytest.mark.parametrize(
    "msg",
    [
        "hello",
        "hi there",
        "what can you do?",
        "/help",
        "how does the agent work?",
        "list available tools",
        "set my researcher preference to terse output",
    ],
)
def test_chitchat_does_not_trigger(msg: str) -> None:
    assert AgentLoop._is_research_conclusion(msg) is False


@pytest.mark.parametrize(
    "msg",
    [
        "Give me the top differentially expressed genes",
        "Run pathway enrichment for these markers",
        "Show genes upregulated in luminal vs basal",
        "rank genes by p-value with fdr<0.05",
        "What is the perturbation distance between A and B?",
        "Return the cluster markers ordered by logfc",
    ],
)
def test_research_questions_trigger(msg: str) -> None:
    assert AgentLoop._is_research_conclusion(msg) is True


def test_appended_hint_keeps_original_reply() -> None:
    head = "I am happy to help once we have data."
    out = AgentLoop._append_evidence_gate(AgentLoop, head, tool_results=None)
    assert out.startswith(head)
    assert "No evidence entries" in out
    assert "census_query" in out
    assert "h5ad" in out


def test_tool_errors_are_listed_in_advisory() -> None:
    out = AgentLoop._append_evidence_gate(
        AgentLoop, "ok", tool_results=["Error: scanpy_de blew up", "OK: not an error"],
    )
    assert "Tool errors during this turn" in out
    assert "scanpy_de blew up" in out
    assert "OK: not an error" not in out

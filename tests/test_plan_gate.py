"""Stage 4.A tests for the PlanGate state machine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from cytopert.agent.loop import (
    PLAN_MODE_AWAITING,
    PLAN_MODE_DISABLED,
    PLAN_MODE_EXECUTING,
    PLAN_MODE_KEY,
    AgentLoop,
    _is_go_phrase,
)
from cytopert.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _CountingProvider(LLMProvider):
    """Records every chat call's tools= argument so tests can assert gating."""

    def __init__(self, replies: list[LLMResponse]) -> None:
        super().__init__(api_key="x")
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
    ) -> LLMResponse:
        self.calls.append({"tools": tools, "messages": messages})
        if not self._replies:
            return LLMResponse(content="...", finish_reason="stop")
        return self._replies.pop(0)

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("go", True),
        ("GO!", True),
        ("execute", True),
        ("approve", True),
        ("yes", True),
        ("go run scanpy", False),  # extra words -> not a bare go signal
        ("nope", False),
        ("", False),
        ("   ", False),
    ],
)
def test_is_go_phrase(phrase: str, expected: bool) -> None:
    assert _is_go_phrase(phrase) is expected


def _build_loop(tmp_path: Path, replies: list[LLMResponse]) -> tuple[AgentLoop, _CountingProvider]:
    provider = _CountingProvider(replies)
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path / "ws",
        model="stub",
        enable_reflection=False,
        load_plugins=False,
    )
    return loop, provider


def test_plan_turn_hides_tools_and_discards_tool_calls(tmp_path: Path) -> None:
    reply = LLMResponse(
        content="Plan: 1) skills_list 2) chains",
        tool_calls=[ToolCallRequest(id="t1", name="skills_list", arguments={})],
        finish_reason="tool_calls",
    )
    loop, prov = _build_loop(tmp_path, [reply])
    loop.enable_plan_gate("ses")
    sess = loop.sessions.get_or_create("ses")
    assert sess.metadata[PLAN_MODE_KEY] == PLAN_MODE_AWAITING

    out = asyncio.run(loop.process_direct("plan a DE run", session_key="ses"))
    assert "PlanGate" in out
    # Provider was called with tools=None during the plan turn.
    assert prov.calls[-1]["tools"] is None


def test_go_phrase_flips_state_and_re_enables_tools(tmp_path: Path) -> None:
    plan_reply = LLMResponse(
        content="Plan: skills_list",
        tool_calls=[ToolCallRequest(id="t1", name="skills_list", arguments={})],
        finish_reason="tool_calls",
    )
    after_go_reply = LLMResponse(content="ok", tool_calls=[], finish_reason="stop")
    loop, prov = _build_loop(tmp_path, [plan_reply, after_go_reply])
    loop.enable_plan_gate("ses")
    asyncio.run(loop.process_direct("plan it", session_key="ses"))
    asyncio.run(loop.process_direct("go", session_key="ses"))
    sess = loop.sessions.get_or_create("ses")
    assert sess.metadata[PLAN_MODE_KEY] == PLAN_MODE_EXECUTING
    assert prov.calls[-1]["tools"] is not None  # tools restored


def test_disabled_plan_mode_passes_tools_through(tmp_path: Path) -> None:
    reply = LLMResponse(content="ok", tool_calls=[], finish_reason="stop")
    loop, prov = _build_loop(tmp_path, [reply])
    sess = loop.sessions.get_or_create("ses")
    sess.metadata[PLAN_MODE_KEY] = PLAN_MODE_DISABLED
    loop.sessions.save(sess)
    asyncio.run(loop.process_direct("hi", session_key="ses"))
    # tools list should be non-empty for the default tool catalog
    assert prov.calls[-1]["tools"]

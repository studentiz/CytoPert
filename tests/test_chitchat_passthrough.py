"""Regression tests covering the bug where every casual message produced
the same Chinese "I need data first" reply.

The fix has three parts that this file pins:

1. ``AgentLoop.process_direct`` must NOT replace the model's reply with
   the evidence-gate hint when the user is just chatting (greeting,
   capability question, gratitude, etc.). The evidence gate is only
   appended for messages that look like a research conclusion request.

2. The interactive shell must NOT silently arm the plan gate -- doing
   so was forcing every turn into "produce a plan, no tools" mode and
   the LLM kept echoing a single "ask for data" template even when the
   user said "hello".

3. The plan-gate slash command must be able to toggle the gate
   explicitly without affecting unrelated session state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cytopert.agent.loop import (
    PLAN_MODE_AWAITING,
    PLAN_MODE_DISABLED,
    PLAN_MODE_KEY,
    AgentLoop,
)
from cytopert.memory.store import MemoryStore
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.evidence_db import EvidenceDB
from cytopert.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from cytopert.skills.manager import SkillsManager


@dataclass
class _ScriptedTurn:
    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"


class _FakeProvider(LLMProvider):
    """Replays a queue of LLMResponses; records the messages it was sent."""

    def __init__(self, script: list[_ScriptedTurn]) -> None:
        super().__init__(api_key="fake", api_base=None)
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls.append(messages)
        if not self._script:
            return LLMResponse(content="(default ok)", finish_reason="stop")
        turn = self._script.pop(0)
        return LLMResponse(
            content=turn.content,
            tool_calls=turn.tool_calls,
            finish_reason=turn.finish_reason,
        )

    def get_default_model(self) -> str:
        return "fake/test"


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def _make_loop(tmp_path: Path, script: list[_ScriptedTurn]) -> tuple[AgentLoop, _FakeProvider]:
    provider = _FakeProvider(script)
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path / "workspace",
        model="fake/test",
        max_iterations=4,
        memory_store=MemoryStore(tmp_path / "memory"),
        skills_manager=SkillsManager(tmp_path / "skills"),
        evidence_db=EvidenceDB(tmp_path / "state.db"),
        chain_store=ChainStore(tmp_path / "state.db", tmp_path / "chains"),
        enable_reflection=False,
        load_plugins=False,
    )
    return loop, provider


@pytest.mark.parametrize(
    "user_input, model_reply",
    [
        ("hello", "Hi! I'm CytoPert. Ask me about a dataset whenever you're ready."),
        ("what can you do?", "I can run scanpy DE, query Census, manage skills, and more."),
        ("Can you discuss something else?", "Sure, what would you like to talk about?"),
        ("thanks!", "You're welcome."),
    ],
)
def test_chitchat_does_not_get_overwritten(
    tmp_path: Path, user_input: str, model_reply: str
) -> None:
    """Regression: greetings + capability questions return the model's text verbatim.

    Previously the evidence-gate / plan-gate combo replaced the reply
    with a fixed "you need data first" template every single turn
    regardless of input. The fix is two-fold: plan-gate is OFF by
    default, and the evidence-gate hint only appends when the user
    actually asks for a reproducible research result.
    """
    loop, provider = _make_loop(
        tmp_path, [_ScriptedTurn(content=model_reply)]
    )
    out = asyncio.run(loop.process_direct(user_input, session_key="chitchat"))
    assert out == model_reply, (
        f"AgentLoop replaced the model reply!\n"
        f"  input    : {user_input!r}\n"
        f"  model    : {model_reply!r}\n"
        f"  returned : {out!r}"
    )


def test_three_consecutive_chitchat_turns_get_distinct_replies(tmp_path: Path) -> None:
    """Pin that distinct LLM replies survive across three turns, never collapsing.

    The original bug produced an identical reply on turn 1 / 2 / 3
    because plan-gate was sticky and the evidence-gate template
    overwrote the LLM's actual output. With both fixes in place each
    scripted reply must reach the user verbatim.
    """
    replies = [
        "Hi there! How can I help with your single-cell project today?",
        "I support scanpy preprocessing, DE, pathway lookup, mechanism chains, and more.",
        "Sure -- what's on your mind?",
    ]
    loop, _ = _make_loop(tmp_path, [_ScriptedTurn(content=r) for r in replies])
    seen: list[str] = []
    for user_input in ["hello", "what can you do?", "Can you discuss something else?"]:
        seen.append(asyncio.run(loop.process_direct(user_input, session_key="chitchat")))
    assert seen == replies
    assert len(set(seen)) == 3, f"Expected 3 distinct replies, got {seen!r}"


def test_plan_gate_is_disabled_by_default(tmp_path: Path) -> None:
    """A brand-new session must NOT come up in awaiting_plan mode."""
    loop, _ = _make_loop(tmp_path, [_ScriptedTurn(content="ack")])
    sess = loop.sessions.get_or_create("ses_default")
    assert (
        sess.metadata.get(PLAN_MODE_KEY, PLAN_MODE_DISABLED) != PLAN_MODE_AWAITING
    ), "Plan-gate should be off by default; the interactive shell opts in via /plan-gate on."


def test_evidence_gate_still_appends_for_research_questions(tmp_path: Path) -> None:
    """The data-request hint must still fire when the user clearly asks for an analysis."""
    model_reply = "I cannot list DE genes without data."
    loop, _ = _make_loop(tmp_path, [_ScriptedTurn(content=model_reply)])
    out = asyncio.run(
        loop.process_direct(
            "Show me the top differentially expressed genes for condition A vs B.",
            session_key="research",
        )
    )
    # Model's reply is preserved verbatim AT THE TOP, then a hint is appended.
    assert out.startswith(model_reply)
    assert "evidence" in out.lower()
    assert "h5ad" in out.lower() or "census_query" in out.lower()


def test_plan_gate_slash_command_arms_and_disarms(tmp_path: Path) -> None:
    """`/plan-gate on` flips PLAN_MODE_KEY to awaiting; `/plan-gate off` to disabled."""
    from rich.console import Console

    from cytopert.cli.interactive_slash import handle_slash_command

    loop, _ = _make_loop(tmp_path, [_ScriptedTurn(content="ack")])
    console = Console(force_terminal=False)

    handle_slash_command("/plan-gate on", loop, "ses", console)
    sess = loop.sessions.get_or_create("ses")
    assert sess.metadata[PLAN_MODE_KEY] == PLAN_MODE_AWAITING

    handle_slash_command("/plan-gate off", loop, "ses", console)
    sess = loop.sessions.get_or_create("ses")
    assert sess.metadata[PLAN_MODE_KEY] == PLAN_MODE_DISABLED

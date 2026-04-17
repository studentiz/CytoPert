"""Integration tests for AgentLoop with the new persistence + memory + skills wiring.

A FakeProvider drives the agent without any real LLM call.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cytopert.agent.loop import AgentLoop
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


class FakeProvider(LLMProvider):
    """Replays a queue of LLMResponses; captures the messages sent in each call."""

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
            return LLMResponse(content="OK", finish_reason="stop")
        turn = self._script.pop(0)
        return LLMResponse(
            content=turn.content,
            tool_calls=turn.tool_calls,
            finish_reason=turn.finish_reason,
        )

    def get_default_model(self) -> str:
        return "fake/test"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Redirect ~/.cytopert to a temp dir for the duration of the test."""
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def _make_loop(tmp_path: Path, script: list[_ScriptedTurn]) -> tuple[AgentLoop, FakeProvider]:
    provider = FakeProvider(script)
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path / "workspace",
        model="fake/test",
        max_iterations=8,
        memory_store=MemoryStore(tmp_path / "memory"),
        skills_manager=SkillsManager(tmp_path / "skills"),
        evidence_db=EvidenceDB(tmp_path / "state.db"),
        chain_store=ChainStore(tmp_path / "state.db", tmp_path / "chains"),
        enable_reflection=False,
    )
    return loop, provider


def test_register_default_tools_includes_new_tools(tmp_path: Path) -> None:
    loop, _ = _make_loop(tmp_path, [_ScriptedTurn(content="done")])
    expected = {"memory", "skills_list", "skill_view", "skill_manage",
                "evidence_search", "chain_status", "chains"}
    assert expected.issubset(set(loop.tools.tool_names))


def test_system_prompt_contains_memory_and_skills(tmp_path: Path) -> None:
    loop, provider = _make_loop(tmp_path, [_ScriptedTurn(content="done")])
    loop.memory.add("context", "Census 2025-11-08 default")
    loop.skills.install_bundled()
    asyncio.run(loop.process_direct("Hi", session_key="t"))
    sys_msg = provider.calls[0][0]
    assert sys_msg["role"] == "system"
    assert "Census 2025-11-08 default" in sys_msg["content"]
    assert "perturbation-de" in sys_msg["content"]
    assert "Memory (frozen snapshot" in sys_msg["content"]


def test_tool_call_persists_evidence(tmp_path: Path) -> None:
    tc = ToolCallRequest(id="tc1", name="scanpy_de",
                         arguments={"path": "x.h5ad", "groupby": "state",
                                    "group1": "A", "group2": "B"})
    script = [
        _ScriptedTurn(tool_calls=[tc], content=""),
        _ScriptedTurn(content="DE done."),
    ]
    loop, _ = _make_loop(tmp_path, script)

    async def fake_execute(name: str, args: dict[str, Any]) -> str:
        if name == "scanpy_de":
            return "DE (A vs B): top genes: ['NFATC1', 'ESR1', 'NOTCH1']"
        return "ok"

    loop.tools.execute = fake_execute
    asyncio.run(loop.process_direct("run DE", session_key="s1"))
    assert loop.evidence_db.count() == 1
    entries = loop.evidence_db.recent(limit=5)
    assert entries[0].tool_name == "scanpy_de"
    assert "NFATC1" in entries[0].genes


def test_chain_tool_persists_to_chain_store(tmp_path: Path) -> None:
    tc = ToolCallRequest(
        id="tc1",
        name="chains",
        arguments={
            "summary": "NFATC1 -> NOTCH -> luminal",
            "evidence_ids": ["e1", "e2"],
            "links": [{"from_node": "NFATC1", "to_node": "NOTCH",
                       "relation": "regulates", "evidence_ids": ["e1"]}],
        },
    )
    script = [
        _ScriptedTurn(tool_calls=[tc], content=""),
        _ScriptedTurn(content="chain submitted"),
    ]
    loop, _ = _make_loop(tmp_path, script)
    asyncio.run(loop.process_direct("propose a chain", session_key="s1"))
    chains = loop.chains.list()
    assert len(chains) == 1
    chain, status = chains[0]
    assert status == "proposed"
    assert chain.summary.startswith("NFATC1")


def test_chain_status_tool_transitions(tmp_path: Path) -> None:
    chain_call = ToolCallRequest(
        id="t1", name="chains",
        arguments={"summary": "x", "evidence_ids": ["e1"]},
    )
    status_call = ToolCallRequest(
        id="t2", name="chain_status",
        arguments={"chain_id": "chain_0001", "status": "supported",
                   "evidence_ids": ["e1"], "note": "exp"},
    )
    script = [
        _ScriptedTurn(tool_calls=[chain_call], content=""),
        _ScriptedTurn(tool_calls=[status_call], content=""),
        _ScriptedTurn(content="updated"),
    ]
    loop, _ = _make_loop(tmp_path, script)
    asyncio.run(loop.process_direct("create then update chain", session_key="s1"))
    assert loop.chains.get_status("chain_0001") == "supported"


def test_evidence_search_tool_returns_persisted_entries(tmp_path: Path) -> None:
    loop, _ = _make_loop(tmp_path, [_ScriptedTurn(content="done")])
    from cytopert.data.models import EvidenceEntry, EvidenceType
    loop.evidence_db.add(
        EvidenceEntry(id="e1", type=EvidenceType.DATA, summary="DE NFATC1 luminal",
                      genes=["NFATC1"], tool_name="scanpy_de"),
        session_id="prev",
    )
    tool = loop.tools.get("evidence_search")
    res = asyncio.run(tool.execute(query="NFATC1"))
    payload = json.loads(res)
    assert payload["success"]
    assert payload["count"] == 1
    assert payload["entries"][0]["id"] == "e1"


def test_memory_tool_changes_persist_for_next_session(tmp_path: Path) -> None:
    loop, provider = _make_loop(tmp_path, [
        _ScriptedTurn(tool_calls=[ToolCallRequest(
            id="m1", name="memory",
            arguments={"action": "add", "target": "researcher",
                       "content": "Prefer concise outputs"})], content=""),
        _ScriptedTurn(content="ok"),
        _ScriptedTurn(content="hi again"),
    ])
    asyncio.run(loop.process_direct("set my prefs", session_key="s1"))
    asyncio.run(loop.process_direct("again", session_key="s1"))
    sys_msg_2 = provider.calls[2][0]["content"]
    assert "Prefer concise outputs" in sys_msg_2


def test_system_prompt_frozen_within_a_single_turn(tmp_path: Path) -> None:
    """Verify acceptance criterion #4: in-session memory edits do NOT rewrite the live
    system prompt — the next LLM call inside the SAME process_direct sees the same
    system message even after the agent ran the `memory` tool to add a new entry.
    """
    add_memory_tc = ToolCallRequest(
        id="t1", name="memory",
        arguments={"action": "add", "target": "context", "content": "INSESSION"},
    )
    script = [
        _ScriptedTurn(tool_calls=[add_memory_tc], content=""),
        _ScriptedTurn(content="all set"),
    ]
    loop, provider = _make_loop(tmp_path, script)
    asyncio.run(loop.process_direct("update memory", session_key="t"))
    assert len(provider.calls) == 2
    sys_first = provider.calls[0][0]["content"]
    sys_second = provider.calls[1][0]["content"]
    assert sys_first == sys_second, "system prompt was rewritten mid-turn"
    assert "INSESSION" not in sys_first
    assert "INSESSION" in loop.memory.read("context")

"""Tests for cytopert.agent.reflection (trigger logic + JSON parsing + apply)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cytopert.agent.loop import AgentLoop
from cytopert.agent.reflection import (
    apply_reflection,
    maybe_reflect,
    parse_reflection_json,
    should_reflect,
)
from cytopert.data.models import MechanismChain
from cytopert.memory.store import MemoryStore
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.evidence_db import EvidenceDB
from cytopert.providers.base import LLMProvider, LLMResponse
from cytopert.skills.manager import SkillsManager


@dataclass
class _Turn:
    content: str | None = None
    finish_reason: str = "stop"


class FakeProvider(LLMProvider):
    def __init__(self, script: list[_Turn]) -> None:
        super().__init__(api_key="fake")
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.calls.append(messages)
        if not self._script:
            return LLMResponse(content="", finish_reason="stop")
        t = self._script.pop(0)
        return LLMResponse(content=t.content, finish_reason=t.finish_reason)

    def get_default_model(self) -> str:
        return "fake"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cyto_home"))


def _make_loop(tmp_path: Path, script: list[_Turn]) -> AgentLoop:
    return AgentLoop(
        provider=FakeProvider(script),
        workspace=tmp_path / "ws",
        model="fake",
        memory_store=MemoryStore(tmp_path / "memory"),
        skills_manager=SkillsManager(tmp_path / "skills"),
        evidence_db=EvidenceDB(tmp_path / "state.db"),
        chain_store=ChainStore(tmp_path / "state.db", tmp_path / "chains"),
        enable_reflection=True,
    )


def test_should_reflect_threshold() -> None:
    assert should_reflect(tool_calls_count=5, chains_touched=[], new_evidence_ids=[])
    assert not should_reflect(tool_calls_count=2, chains_touched=[], new_evidence_ids=[])
    assert should_reflect(tool_calls_count=1, chains_touched=["c1"], new_evidence_ids=[])
    assert should_reflect(tool_calls_count=1, chains_touched=[], new_evidence_ids=["a", "b", "c"])
    assert should_reflect(tool_calls_count=0, chains_touched=[], new_evidence_ids=[],
                          user_feedback="exp X refuted")


def test_parse_reflection_json_plain() -> None:
    out = parse_reflection_json('{"memory_updates": [], "skill_proposals": [], "chain_status_updates": []}')
    assert out == {"memory_updates": [], "skill_proposals": [], "chain_status_updates": []}


def test_parse_reflection_json_fenced() -> None:
    text = "thinking...\n```json\n{\"memory_updates\": [{\"action\": \"add\"}]}\n```\nthat's it"
    out = parse_reflection_json(text)
    assert out["memory_updates"] == [{"action": "add"}]


def test_parse_reflection_json_garbage() -> None:
    assert parse_reflection_json(None) == {}
    assert parse_reflection_json("no json here") == {}


def test_apply_reflection_writes_memory_and_stages_skill(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, [_Turn(content="ignored")])
    payload = {
        "memory_updates": [
            {"action": "add", "target": "context", "content": "Census 2025-11-08 default"},
            {"action": "add", "target": "researcher", "content": "Concise summaries"},
        ],
        "skill_proposals": [
            {"name": "auto-skill", "category": "pipelines", "description": "auto",
             "content": "---\nname: auto-skill\ndescription: auto\n---\n# body"},
        ],
        "chain_status_updates": [],
    }
    summary = apply_reflection(loop, payload)
    assert summary["memory_applied"] == 2
    assert summary["skills_staged"] == 1
    assert "Census 2025-11-08" in loop.memory.read("context")
    staged = [s for s in loop.skills.list(include_staged=True) if s.name == "auto-skill"]
    assert staged and staged[0].staged is True


def test_apply_reflection_chain_update(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, [_Turn(content="ignored")])
    loop.chains.upsert(MechanismChain(id="chain_1", summary="s", evidence_ids=["e1"]),
                       status="proposed")
    payload = {
        "memory_updates": [],
        "skill_proposals": [],
        "chain_status_updates": [
            {"chain_id": "chain_1", "status": "supported",
             "evidence_ids": ["e2"], "note": "exp confirmed"},
        ],
    }
    summary = apply_reflection(loop, payload)
    assert summary["chains_updated"] == 1
    assert loop.chains.get_status("chain_1") == "supported"


def test_apply_reflection_records_errors(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, [_Turn(content="ignored")])
    payload = {
        "memory_updates": [{"action": "add", "target": "context", "content": ""}],
        "skill_proposals": [{"name": "Bad NAME", "content": "x"}],
        "chain_status_updates": [{"chain_id": "missing", "status": "supported"}],
    }
    summary = apply_reflection(loop, payload)
    assert summary["memory_applied"] == 0
    assert summary["skills_staged"] == 0
    assert summary["chains_updated"] == 0
    assert len(summary["errors"]) >= 3


def test_maybe_reflect_skips_when_not_triggered(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, [_Turn(content='{"memory_updates":[]}')])
    out = asyncio.run(maybe_reflect(
        loop=loop, session_key="s",
        user_message="hi", final_response="hello",
        tool_calls_count=0, chains_touched=[], new_evidence_ids=[],
    ))
    assert out is None


def test_maybe_reflect_runs_and_applies(tmp_path: Path) -> None:
    payload = json.dumps({
        "memory_updates": [{"action": "add", "target": "context",
                             "content": "Default census 2025-11-08"}],
        "skill_proposals": [],
        "chain_status_updates": [],
    })
    loop = _make_loop(tmp_path, [_Turn(content=payload)])
    out = asyncio.run(maybe_reflect(
        loop=loop, session_key="s",
        user_message="run pipeline", final_response="done",
        tool_calls_count=6, chains_touched=[], new_evidence_ids=[],
    ))
    assert out is not None
    assert out["memory_applied"] == 1
    assert "Default census" in loop.memory.read("context")


def test_maybe_reflect_handles_invalid_json(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, [_Turn(content="this is not json")])
    out = asyncio.run(maybe_reflect(
        loop=loop, session_key="s",
        user_message="x", final_response="y",
        tool_calls_count=10, chains_touched=[], new_evidence_ids=[],
    ))
    assert out is None

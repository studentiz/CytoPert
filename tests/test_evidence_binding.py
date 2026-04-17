"""Stage 5 tests for the evidence-binding enforcer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from cytopert.agent.loop import AgentLoop, _extract_evidence_citations
from cytopert.data.models import EvidenceEntry, EvidenceType
from cytopert.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _ScriptedProvider(LLMProvider):
    """Returns each scripted reply in order; raises if the script runs out."""

    def __init__(self, replies: list[LLMResponse]) -> None:
        super().__init__(api_key="x")
        self._replies = list(replies)
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if not self._replies:
            return LLMResponse(content="...", finish_reason="stop")
        return self._replies.pop(0)

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def test_extractor_handles_brackets_parens_and_dedupes() -> None:
    text = (
        "see [evidence: tool_a, tool_b] and (evidence: tool_a) plus "
        "[evidence: phantom_xx]."
    )
    assert _extract_evidence_citations(text) == ["tool_a", "tool_b", "phantom_xx"]


def test_no_citations_means_no_retry(tmp_path: Path) -> None:
    reply = LLMResponse(
        content="Hi, I'm CytoPert.", tool_calls=[], finish_reason="stop"
    )
    prov = _ScriptedProvider([reply])
    loop = AgentLoop(
        provider=prov, workspace=tmp_path / "ws", model="stub",
        enable_reflection=False, load_plugins=False,
    )
    asyncio.run(loop.process_direct("hello", session_key="ses"))
    assert prov.calls == 1


def test_phantom_citation_triggers_one_retry_that_fixes_it(tmp_path: Path) -> None:
    bad = LLMResponse(
        content="Conclusion [evidence: tool_real, phantom_xx].",
        finish_reason="stop",
    )
    good = LLMResponse(
        content="Conclusion [evidence: tool_real].",
        finish_reason="stop",
    )
    prov = _ScriptedProvider([bad, good])
    loop = AgentLoop(
        provider=prov, workspace=tmp_path / "ws", model="stub",
        enable_reflection=False, load_plugins=False,
    )
    loop.evidence_db.add(
        EvidenceEntry(id="tool_real", type=EvidenceType.DATA,
                      summary="seed", tool_name="scanpy_de"),
        session_id="seed",
    )
    loop._evidence_store.append(loop.evidence_db.get("tool_real"))
    out = asyncio.run(loop.process_direct(
        "Give me top genes upregulated in luminal", session_key="ses",
    ))
    assert prov.calls == 2
    assert "phantom_xx" not in out


def test_persistent_phantom_appends_advisory(tmp_path: Path) -> None:
    bad = LLMResponse(
        content="Conclusion [evidence: phantom_only].", finish_reason="stop"
    )
    still_bad = LLMResponse(
        content="Still phantom [evidence: phantom_only].", finish_reason="stop"
    )
    prov = _ScriptedProvider([bad, still_bad])
    loop = AgentLoop(
        provider=prov, workspace=tmp_path / "ws", model="stub",
        enable_reflection=False, load_plugins=False,
    )
    out = asyncio.run(loop.process_direct(
        "give me top genes upregulated", session_key="ses",
    ))
    assert prov.calls == 2
    assert "[Evidence binding]" in out
    assert "phantom_only" in out

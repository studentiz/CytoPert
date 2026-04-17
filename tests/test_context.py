"""Tests for ContextBuilder memory + skills injection."""

from pathlib import Path

from cytopert.agent.context import ContextBuilder
from cytopert.memory.store import MemoryStore
from cytopert.skills.manager import SkillsManager


def test_build_system_prompt_no_optional_blocks(tmp_path: Path) -> None:
    cb = ContextBuilder(tmp_path)
    prompt = cb.build_system_prompt()
    assert "CytoPert" in prompt
    assert "Memory (frozen snapshot" not in prompt
    assert "Skills (Level 0 index)" not in prompt


def test_build_system_prompt_with_memory_and_skills(tmp_path: Path) -> None:
    cb = ContextBuilder(tmp_path)
    mem = MemoryStore(tmp_path / "memory", limits={"context": 1000, "researcher": 500, "hypothesis_log": 1000})
    mem.add("context", "Use Census 2025-11-08 by default")
    mem.add("researcher", "Prefer concise mechanism summaries")
    skills = SkillsManager(tmp_path / "skills")
    skills.install_bundled()
    prompt = cb.build_system_prompt(
        memory_snapshot=mem.render_snapshot(),
        skills_index=skills.render_index(),
        evidence_summary="- [e1] (data) DE basal vs luminal",
    )
    assert "Memory (frozen snapshot" in prompt
    assert "Census 2025-11-08" in prompt
    assert "Skills (Level 0 index)" in prompt
    assert "perturbation-de" in prompt
    assert "Evidence Summary" in prompt
    assert "Constraints" in prompt


def test_build_messages_uses_current_message(tmp_path: Path) -> None:
    cb = ContextBuilder(tmp_path)
    msgs = cb.build_messages(history=[{"role": "user", "content": "old"}],
                             current_message="hello")
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "hello"}
    assert any(m["content"] == "old" for m in msgs)


def test_self_improvement_section_in_identity(tmp_path: Path) -> None:
    cb = ContextBuilder(tmp_path)
    prompt = cb.build_system_prompt()
    assert "Self-Improvement Loop" in prompt
    assert "evidence_search" in prompt
    assert "chain_status" in prompt

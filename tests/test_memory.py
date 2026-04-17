"""Tests for cytopert.memory.store / tool."""

import asyncio
import json
from pathlib import Path

import pytest

from cytopert.memory.store import MEMORY_TARGETS, MemoryStore, sanitize_entry
from cytopert.memory.tool import MemoryTool


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory", limits={"context": 200, "researcher": 100, "hypothesis_log": 300})


def test_targets_constant() -> None:
    assert set(MEMORY_TARGETS) == {"context", "researcher", "hypothesis_log"}


def test_add_and_read(store: MemoryStore) -> None:
    res = store.add("context", "User runs Census 2025-11-08 by default")
    assert res.success
    assert res.entries == ["User runs Census 2025-11-08 by default"]
    assert res.usage_chars > 0
    assert "Census" in store.read("context")


def test_dedupe(store: MemoryStore) -> None:
    store.add("researcher", "Prefer concise mechanism summaries")
    res = store.add("researcher", "Prefer concise mechanism summaries")
    assert res.success and "Duplicate" in res.message
    assert len(store.entries("researcher")) == 1


def test_limit_enforced(store: MemoryStore) -> None:
    store.add("researcher", "x" * 50)
    res = store.add("researcher", "y" * 90)
    assert not res.success
    assert "exceed" in res.message.lower()


def test_replace_substring(store: MemoryStore) -> None:
    store.add("context", "Default scanpy min_genes=200")
    res = store.replace("context", "min_genes=200", "Default scanpy min_genes=500 (per researcher request)")
    assert res.success, res.message
    assert any("min_genes=500" in e for e in store.entries("context"))


def test_replace_ambiguous(store: MemoryStore) -> None:
    store.add("context", "scanpy default A")
    store.add("context", "scanpy default B")
    res = store.replace("context", "scanpy default", "merged")
    assert not res.success
    assert "matched 2 entries" in res.message


def test_remove(store: MemoryStore) -> None:
    store.add("hypothesis_log", "chain_0001 NFATC1->NOTCH proposed")
    store.add("hypothesis_log", "chain_0002 ESR1->WNT supported")
    res = store.remove("hypothesis_log", "chain_0001")
    assert res.success
    assert len(store.entries("hypothesis_log")) == 1
    assert store.entries("hypothesis_log")[0].startswith("chain_0002")


def test_remove_no_match(store: MemoryStore) -> None:
    res = store.remove("context", "nothing")
    assert not res.success


def test_render_snapshot_includes_all_targets(store: MemoryStore) -> None:
    store.add("context", "ctx entry")
    store.add("researcher", "Prefers Chinese output")
    snapshot = store.render_snapshot()
    assert "CONTEXT" in snapshot
    assert "RESEARCHER" in snapshot
    assert "HYPOTHESIS LOG" in snapshot
    assert "(empty)" in snapshot
    assert "ctx entry" in snapshot


def test_invalid_target_raises(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        store.add("notes", "x")


def test_clear_specific_target(store: MemoryStore) -> None:
    store.add("context", "a")
    store.add("researcher", "b")
    store.clear("context")
    assert store.entries("context") == []
    assert store.entries("researcher") == ["b"]


def test_sanitize_entry_strips_invisibles() -> None:
    text = "User\u200bprefers\u202fcompact"
    cleaned = sanitize_entry(text)
    assert "\u200b" not in cleaned and "\u202f" not in cleaned


def test_memory_tool_add_and_replace(store: MemoryStore) -> None:
    tool = MemoryTool(store)
    res = asyncio.run(tool.execute(action="add", target="context", content="hello"))
    payload = json.loads(res)
    assert payload["success"]

    res2 = asyncio.run(tool.execute(action="replace", target="context",
                                    content="hello world", old_text="hello"))
    payload2 = json.loads(res2)
    assert payload2["success"]
    assert any("hello world" in e for e in store.entries("context"))


def test_memory_tool_unknown_action(store: MemoryStore) -> None:
    tool = MemoryTool(store)
    res = asyncio.run(tool.execute(action="zzz", target="context"))
    payload = json.loads(res)
    assert not payload["success"]


def test_memory_tool_invalid_target_handled(store: MemoryStore) -> None:
    tool = MemoryTool(store)
    res = asyncio.run(tool.execute(action="add", target="notes", content="x"))
    payload = json.loads(res)
    assert not payload["success"]

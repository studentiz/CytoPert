"""Tests for cytopert.skills.manager / tool."""

import asyncio
import json
from pathlib import Path

import pytest

from cytopert.skills.manager import SkillsManager, parse_frontmatter
from cytopert.skills.tool import SkillManageTool, SkillsListTool, SkillViewTool


@pytest.fixture
def manager(tmp_path: Path) -> SkillsManager:
    return SkillsManager(tmp_path / "skills")


def test_install_bundled(manager: SkillsManager) -> None:
    n = manager.install_bundled()
    assert n == 3
    skills = manager.list()
    names = {s.name for s in skills}
    assert names == {"perturbation-de", "census-tissue-slice", "mechanism-chain-from-de"}
    cats = {s.category for s in skills}
    assert cats == {"pipelines", "reasoning"}


def test_install_bundled_skips_when_manifest_present(manager: SkillsManager) -> None:
    n1 = manager.install_bundled()
    n2 = manager.install_bundled()
    assert n1 == 3 and n2 == 0


def test_view_returns_full_content(manager: SkillsManager) -> None:
    manager.install_bundled()
    body = manager.view("perturbation-de")
    assert body.startswith("---")
    assert "Perturbation DE" in body


def test_view_unknown_raises(manager: SkillsManager) -> None:
    with pytest.raises(FileNotFoundError):
        manager.view("nope-skill")


def test_render_index_only_returns_level0(manager: SkillsManager) -> None:
    manager.install_bundled()
    idx = manager.render_index()
    assert "perturbation-de" in idx
    assert "## Procedure" not in idx
    assert idx.count("\n") == 2


def test_render_index_filters_by_available_tools(manager: SkillsManager) -> None:
    manager.install_bundled()
    idx_full = manager.render_index(available_tools={"census_query", "scanpy_preprocess",
                                                     "scanpy_de", "chains", "chain_status",
                                                     "evidence"})
    assert "perturbation-de" in idx_full
    idx_partial = manager.render_index(available_tools={"census_query"})
    assert "census-tissue-slice" in idx_partial
    assert "perturbation-de" not in idx_partial


def test_create_in_staged_by_default(manager: SkillsManager) -> None:
    content = (
        "---\nname: my-flow\ndescription: Test skill\nmetadata:\n  cytopert:\n"
        "    category: pipelines\n---\n# My Flow\n## When to Use\nWhenever\n"
    )
    path = manager.create("my-flow", content, category="pipelines", staged=True)
    assert ".staged" in str(path)
    skills_live = manager.list(include_staged=False)
    assert "my-flow" not in {s.name for s in skills_live}
    skills_all = manager.list(include_staged=True)
    assert any(s.name == "my-flow" and s.staged for s in skills_all)


def test_create_live_when_staged_false(manager: SkillsManager) -> None:
    content = "---\nname: live-skill\ndescription: Test\n---\n# body\n"
    manager.create("live-skill", content, category="pipelines", staged=False)
    assert "live-skill" in {s.name for s in manager.list()}


def test_accept_staged_promotes(manager: SkillsManager) -> None:
    content = (
        "---\nname: promote-me\ndescription: Test\nmetadata:\n  cytopert:\n"
        "    category: reasoning\n---\n# body\n"
    )
    manager.create("promote-me", content, category="reasoning", staged=True)
    target = manager.accept_staged("promote-me")
    assert target.is_file()
    live = {s.name for s in manager.list()}
    assert "promote-me" in live


def test_patch_skill(manager: SkillsManager) -> None:
    manager.install_bundled()
    manager.patch("perturbation-de", "## When to Use", "## When to Use (updated)")
    body = manager.view("perturbation-de")
    assert "## When to Use (updated)" in body


def test_patch_ambiguous_raises(manager: SkillsManager) -> None:
    manager.install_bundled()
    with pytest.raises(ValueError):
        manager.patch("perturbation-de", "## ", "X")


def test_delete(manager: SkillsManager) -> None:
    manager.install_bundled()
    manager.delete("census-tissue-slice")
    assert "census-tissue-slice" not in {s.name for s in manager.list()}


def test_invalid_name_rejected(manager: SkillsManager) -> None:
    with pytest.raises(ValueError):
        manager.create("Bad NAME", "---\nname: bad\n---\n# body", category="pipelines")


def test_view_file_path_traversal_blocked(manager: SkillsManager) -> None:
    manager.install_bundled()
    with pytest.raises((PermissionError, FileNotFoundError)):
        manager.view_file("perturbation-de", "../../../etc/passwd")


def test_parse_frontmatter_valid_and_invalid() -> None:
    meta, body = parse_frontmatter("---\nname: x\n---\nhi\n")
    assert meta == {"name": "x"} and "hi" in body
    meta2, body2 = parse_frontmatter("no frontmatter")
    assert meta2 == {} and body2 == "no frontmatter"


def test_skills_list_tool(manager: SkillsManager) -> None:
    manager.install_bundled()
    tool = SkillsListTool(manager)
    payload = json.loads(asyncio.run(tool.execute()))
    assert any(s["name"] == "perturbation-de" for s in payload)


def test_skill_view_tool(manager: SkillsManager) -> None:
    manager.install_bundled()
    tool = SkillViewTool(manager)
    body = asyncio.run(tool.execute(name="perturbation-de"))
    assert "Procedure" in body


def test_skill_manage_create_then_accept(manager: SkillsManager) -> None:
    tool = SkillManageTool(manager)
    res = json.loads(asyncio.run(tool.execute(
        action="create", name="test-pipe",
        content="---\nname: test-pipe\ndescription: t\n---\n# x",
        category="pipelines", staged=True,
    )))
    assert res["success"]
    accept = json.loads(asyncio.run(tool.execute(
        action="accept_staged", name="test-pipe", category="pipelines",
    )))
    assert accept["success"]
    assert "test-pipe" in {s.name for s in manager.list()}


def test_skill_manage_patch_error_propagates(manager: SkillsManager) -> None:
    tool = SkillManageTool(manager)
    res = json.loads(asyncio.run(tool.execute(
        action="patch", name="missing-skill",
        old_string="x", new_string="y",
    )))
    assert not res["success"]

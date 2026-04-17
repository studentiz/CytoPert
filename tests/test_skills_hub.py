"""Stage 13b tests: skills hub install / search / uninstall."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cytopert.cli.commands import app
from cytopert.skills.hub import install_from_source
from cytopert.skills.manager import SkillsManager


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    fake_root = tmp_path / "cytopert_root"
    fake_root.mkdir()
    monkeypatch.delenv("CYTOPERT_HOME", raising=False)
    import cytopert.utils.helpers as hh

    monkeypatch.setattr(hh, "CYTOPERT_ROOT_DIR", fake_root)
    yield


def _runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


def _make_skill_tree(root: Path, name: str = "demo-skill") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Demo skill for tests.\n"
        "version: 0.1.0\n"
        "metadata:\n"
        "  cytopert:\n"
        "    category: user\n"
        "    tags: [demo]\n"
        "---\n"
        "# demo\n",
        encoding="utf-8",
    )
    return folder


def test_install_from_directory(tmp_path: Path) -> None:
    src = _make_skill_tree(tmp_path / "src")
    mgr = SkillsManager(tmp_path / "skills")
    out = install_from_source(mgr, source=str(src), category="user")
    assert out.is_file() and out.name == "SKILL.md"
    assert (mgr.skills_dir / "user" / "demo-skill" / "SKILL.md").exists()


def test_install_rejects_directory_without_skill_md(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    mgr = SkillsManager(tmp_path / "skills")
    with pytest.raises(FileNotFoundError):
        install_from_source(mgr, source=str(empty))


def test_install_from_zip(tmp_path: Path) -> None:
    src = _make_skill_tree(tmp_path / "src", name="zipme")
    archive = tmp_path / "zipme.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in src.rglob("*"):
            z.write(path, arcname=path.relative_to(src.parent))
    mgr = SkillsManager(tmp_path / "skills")
    out = install_from_source(mgr, source=str(archive), category="user")
    assert out.exists()
    assert (mgr.skills_dir / "user" / "zipme" / "SKILL.md").is_file()


def test_install_from_tar_gz(tmp_path: Path) -> None:
    src = _make_skill_tree(tmp_path / "src", name="tarball")
    archive = tmp_path / "tarball.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        t.add(src, arcname="tarball")
    mgr = SkillsManager(tmp_path / "skills")
    out = install_from_source(mgr, source=str(archive), category="user")
    assert out.exists()


def test_install_force_overwrites(tmp_path: Path) -> None:
    src = _make_skill_tree(tmp_path / "src", name="overwrite-me")
    mgr = SkillsManager(tmp_path / "skills")
    install_from_source(mgr, source=str(src))
    # Modify source then re-install with --force
    (src / "SKILL.md").write_text(
        "---\nname: overwrite-me\ndescription: v2\nversion: 0.2.0\n---\n# new",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        install_from_source(mgr, source=str(src))  # no force
    install_from_source(mgr, source=str(src), force=True)
    text = (mgr.skills_dir / "user" / "overwrite-me" / "SKILL.md").read_text("utf-8")
    assert "v2" in text


def test_cli_search_matches_name_and_description() -> None:
    runner = _runner()
    # The bundled skills get installed by SkillsManager; ensure at least
    # one search hit by adding a skill via the CLI install path.
    import cytopert.utils.helpers as hh

    src = _make_skill_tree(hh.CYTOPERT_ROOT_DIR / "src", name="searchable-bot")
    res = runner.invoke(app, ["skills", "install", str(src)])
    assert res.exit_code == 0, res.stdout + res.stderr
    hit = runner.invoke(app, ["skills", "search", "searchable"])
    assert hit.exit_code == 0
    assert "searchable-bot" in hit.stdout
    miss = runner.invoke(app, ["skills", "search", "no-such-needle-12345"])
    assert miss.exit_code == 0
    assert "No skills matched" in miss.stdout


def test_cli_uninstall_removes_skill() -> None:
    runner = _runner()
    import cytopert.utils.helpers as hh

    src = _make_skill_tree(hh.CYTOPERT_ROOT_DIR / "src", name="goodbye-skill")
    inst = runner.invoke(app, ["skills", "install", str(src)])
    assert inst.exit_code == 0
    rm = runner.invoke(app, ["skills", "uninstall", "goodbye-skill", "--yes"])
    assert rm.exit_code == 0, rm.stdout + rm.stderr
    listed = runner.invoke(app, ["skills", "list"])
    assert "goodbye-skill" not in listed.stdout

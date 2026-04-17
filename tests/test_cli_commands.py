"""Smoke tests for the new CytoPert CLI subcommands using Typer's runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cytopert.cli.commands import app
from cytopert.data.models import EvidenceEntry, EvidenceType, MechanismChain
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.evidence_db import EvidenceDB


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cyto"))
    yield


def _runner() -> CliRunner:
    return CliRunner()


def test_memory_show_empty() -> None:
    result = _runner().invoke(app, ["memory", "show"])
    assert result.exit_code == 0
    assert "(empty)" in result.stdout


def test_memory_clear_with_yes() -> None:
    from cytopert.memory.store import MemoryStore
    from cytopert.utils.helpers import get_memory_dir

    store = MemoryStore(get_memory_dir())
    store.add("context", "x")
    res = _runner().invoke(app, ["memory", "clear", "--target", "context", "--yes"])
    assert res.exit_code == 0
    assert "Cleared" in res.stdout


def test_skills_list_empty() -> None:
    result = _runner().invoke(app, ["skills", "list"])
    assert result.exit_code == 0


def test_skills_show_after_install(tmp_path: Path) -> None:
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import get_skills_dir

    SkillsManager(get_skills_dir()).install_bundled()
    res_list = _runner().invoke(app, ["skills", "list"])
    assert "perturbation-de" in res_list.stdout
    res_show = _runner().invoke(app, ["skills", "show", "perturbation-de"])
    assert res_show.exit_code == 0
    assert "Procedure" in res_show.stdout


def test_skills_new_then_accept() -> None:
    res_new = _runner().invoke(app, ["skills", "new", "my-pipe", "-c", "pipelines",
                                     "-d", "test", "--staged"])
    assert res_new.exit_code == 0, res_new.stdout
    res_accept = _runner().invoke(app, ["skills", "accept", "my-pipe", "-c", "pipelines"])
    assert res_accept.exit_code == 0, res_accept.stdout
    res_list = _runner().invoke(app, ["skills", "list"])
    assert "my-pipe" in res_list.stdout


def test_chains_list_empty_and_show_missing() -> None:
    res_list = _runner().invoke(app, ["chains", "list"])
    assert res_list.exit_code == 0
    res_show = _runner().invoke(app, ["chains", "show", "nope"])
    assert res_show.exit_code == 1


def test_chains_list_and_show_after_seed() -> None:
    from cytopert.utils.helpers import get_chains_dir, get_state_db_path

    store = ChainStore(get_state_db_path(), get_chains_dir())
    chain = MechanismChain(id="chain_demo", summary="NFATC1 -> NOTCH",
                           evidence_ids=["e1"])
    store.upsert(chain, status="proposed")
    res = _runner().invoke(app, ["chains", "list"])
    assert "chain_demo" in res.stdout
    res2 = _runner().invoke(app, ["chains", "show", "chain_demo"])
    assert res2.exit_code == 0
    assert "NFATC1" in res2.stdout


def test_evidence_search_returns_seeded() -> None:
    from cytopert.utils.helpers import get_state_db_path

    db = EvidenceDB(get_state_db_path())
    db.add(EvidenceEntry(id="e1", type=EvidenceType.DATA, summary="DE NFATC1 luminal",
                         genes=["NFATC1"], tool_name="scanpy_de"))
    res = _runner().invoke(app, ["evidence", "search", "NFATC1"])
    assert res.exit_code == 0
    assert "e1" in res.stdout
    res2 = _runner().invoke(app, ["evidence", "show", "e1"])
    assert res2.exit_code == 0
    assert "NFATC1" in res2.stdout


def test_status_command_runs() -> None:
    res = _runner().invoke(app, ["status"])
    assert res.exit_code == 0
    assert "CytoPert Status" in res.stdout
    assert "Memory[context]" in res.stdout

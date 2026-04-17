"""Stage 7.4 tests for the trajectory writer."""

from __future__ import annotations

import json

import pytest

from cytopert.agent.trajectory import (
    convert_session_to_sharegpt,
    save_trajectory,
    trajectories_dir,
)
from cytopert.session.manager import Session


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def test_convert_session_to_sharegpt_drops_system_by_default() -> None:
    sess = Session(key="t")
    sess.add_message("system", "hidden")
    sess.add_message("user", "hi")
    sess.add_message("assistant", "hello")
    conv = convert_session_to_sharegpt(sess)
    assert conv == [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "hello"},
    ]


def test_convert_session_to_sharegpt_can_include_system() -> None:
    sess = Session(key="t")
    sess.add_message("system", "S")
    sess.add_message("user", "u")
    conv = convert_session_to_sharegpt(sess, include_system=True)
    assert conv[0] == {"from": "system", "value": "S"}


def test_save_trajectory_writes_jsonl_with_metadata() -> None:
    conv = [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "hello"},
    ]
    out = save_trajectory(
        conv,
        model="stub",
        completed=True,
        evidence_ids=["tool_x_1"],
        chains_touched=["chain_0001"],
        session_key="t1",
    )
    assert out.exists()
    assert out.parent == trajectories_dir()
    line = out.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["evidence_ids"] == ["tool_x_1"]
    assert payload["chains_touched"] == ["chain_0001"]
    assert payload["session_key"] == "t1"
    assert payload["completed"] is True
    assert payload["conversations"] == conv


def test_failed_completion_routes_to_failed_trajectories() -> None:
    out = save_trajectory(
        [{"from": "human", "value": "x"}],
        model="stub",
        completed=False,
    )
    assert out.name == "failed_trajectories.jsonl"


def test_explicit_filename_override(tmp_path) -> None:
    target = tmp_path / "elsewhere" / "t.jsonl"
    out = save_trajectory(
        [{"from": "human", "value": "x"}],
        model="stub",
        completed=True,
        filename=target,
    )
    assert out == target.resolve()
    assert out.exists()

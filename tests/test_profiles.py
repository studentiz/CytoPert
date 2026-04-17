"""Stage 12 tests for profile isolation."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cytopert.cli.commands import app


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Pin CYTOPERT_ROOT to a tempdir so the profile machinery is hermetic.

    The profile system reads ``CYTOPERT_ROOT_DIR`` and ``profiles_dir()``
    helpers from cytopert.utils.helpers; we monkey-patch the constants
    so each test gets its own clean tree.
    """
    fake_root = tmp_path / "cytopert_root"
    fake_root.mkdir()
    monkeypatch.delenv("CYTOPERT_HOME", raising=False)
    import cytopert.utils.helpers as hh

    monkeypatch.setattr(hh, "CYTOPERT_ROOT_DIR", fake_root)
    yield


def _runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


def test_profile_lifecycle_new_use_show_list_delete(monkeypatch) -> None:
    runner = _runner()

    # Initially no profile.
    result = runner.invoke(app, ["profile", "show"])
    assert result.exit_code == 0
    assert "No active profile" in result.stdout

    # Create two named profiles.
    assert runner.invoke(app, ["profile", "new", "studyA"]).exit_code == 0
    assert runner.invoke(app, ["profile", "new", "studyB"]).exit_code == 0

    listed = runner.invoke(app, ["profile", "list"])
    assert listed.exit_code == 0
    assert "studyA" in listed.stdout and "studyB" in listed.stdout

    # Activate one.
    used = runner.invoke(app, ["profile", "use", "studyA"])
    assert used.exit_code == 0
    show2 = runner.invoke(app, ["profile", "show"])
    assert "studyA" in show2.stdout

    # Delete the active one; the active marker should clear automatically.
    deleted = runner.invoke(app, ["profile", "delete", "studyA", "--yes"])
    assert deleted.exit_code == 0
    show3 = runner.invoke(app, ["profile", "show"])
    assert "No active profile" in show3.stdout


def test_dash_p_flag_writes_into_profile_specific_root(tmp_path, monkeypatch) -> None:
    """`cytopert -p name config set ...` lands in profiles/<name>/config.json."""
    runner = _runner()

    # Create the profile so the env-var override lands on a real dir.
    assert runner.invoke(app, ["profile", "new", "edits"]).exit_code == 0

    res = runner.invoke(
        app, ["-p", "edits", "config", "set", "agents.defaults.temperature", "0.42"]
    )
    assert res.exit_code == 0, res.stdout + res.stderr

    import cytopert.utils.helpers as hh

    profile_cfg = (
        hh.CYTOPERT_ROOT_DIR / hh.PROFILES_SUBDIR / "edits" / "config.json"
    )
    default_cfg = hh.CYTOPERT_ROOT_DIR / "config.json"
    assert profile_cfg.exists(), "expected per-profile config.json"
    assert not default_cfg.exists(), "default root config.json must not be touched"
    payload = json.loads(profile_cfg.read_text(encoding="utf-8"))
    assert payload["agents"]["defaults"]["temperature"] == 0.42


def test_active_profile_routes_get_data_path(monkeypatch) -> None:
    """get_data_path follows the active_profile file when env var is unset."""
    runner = _runner()
    assert runner.invoke(app, ["profile", "new", "rox"]).exit_code == 0
    assert runner.invoke(app, ["profile", "use", "rox"]).exit_code == 0

    # Re-import nothing -- the resolver is dynamic; just call it.
    from cytopert.utils.helpers import (
        CYTOPERT_ROOT_DIR,
        PROFILES_SUBDIR,
        get_data_path,
    )

    home = get_data_path()
    assert home == CYTOPERT_ROOT_DIR / PROFILES_SUBDIR / "rox"


def test_env_var_override_wins_over_active_profile(tmp_path, monkeypatch) -> None:
    runner = _runner()
    assert runner.invoke(app, ["profile", "new", "loser"]).exit_code == 0
    assert runner.invoke(app, ["profile", "use", "loser"]).exit_code == 0

    # Now set CYTOPERT_HOME explicitly to a different directory; the
    # explicit override must win over the active_profile file.
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    monkeypatch.setenv("CYTOPERT_HOME", str(explicit_dir))

    from cytopert.utils.helpers import get_data_path

    assert get_data_path() == explicit_dir

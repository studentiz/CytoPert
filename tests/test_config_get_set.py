"""Stage 10.C regression tests for cytopert config get/set."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cytopert.cli.commands import app


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def _runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


def _seed_config(tmp_path) -> None:
    from cytopert.config.loader import save_config
    from cytopert.config.schema import (
        AgentDefaults,
        AgentsConfig,
        Config,
        ProviderConfig,
        ProvidersConfig,
    )

    cfg = Config(
        providers=ProvidersConfig(deepseek=ProviderConfig(api_key="sk-seed")),
        agents=AgentsConfig(defaults=AgentDefaults(model="deepseek-chat")),
    )
    save_config(cfg)


def test_config_get_reports_existing_value(tmp_path) -> None:
    _seed_config(tmp_path)
    runner = _runner()
    result = runner.invoke(app, ["config", "get", "agents.defaults.model"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "deepseek-chat" in result.stdout


def test_config_set_persists_value_and_validates(tmp_path) -> None:
    _seed_config(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app, ["config", "set", "agents.defaults.temperature", "0.4"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    # Verify on disk.
    from cytopert.config.loader import get_config_path
    data = json.loads(get_config_path().read_text(encoding="utf-8"))
    assert data["agents"]["defaults"]["temperature"] == 0.4


def test_config_set_rejects_invalid_path(tmp_path) -> None:
    _seed_config(tmp_path)
    runner = _runner()
    # Trailing dots produce an empty segment; should be rejected.
    result = runner.invoke(app, ["config", "set", "...", "1"])
    assert result.exit_code != 0
    assert "non-empty" in (result.stdout + result.stderr).lower()


def test_config_get_missing_key_exits_nonzero(tmp_path) -> None:
    _seed_config(tmp_path)
    runner = _runner()
    result = runner.invoke(app, ["config", "get", "nope.does.not.exist"])
    assert result.exit_code != 0
    assert "no such key" in (result.stdout + result.stderr).lower()

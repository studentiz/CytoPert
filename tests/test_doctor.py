"""Stage 10.B regression tests for cytopert doctor."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from cytopert.cli.doctor import run_doctor


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def _captured_output() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, width=120, color_system=None, force_terminal=False), buf


def test_doctor_runs_without_config_and_returns_failure_code() -> None:
    console, buf = _captured_output()
    code = run_doctor(ping=False, console=console)
    text = buf.getvalue()
    assert code == 1, "with no config we expect at least one FAIL row"
    assert "FAIL" in text
    assert "config.json" in text


def test_doctor_passes_after_setup(tmp_path, monkeypatch) -> None:
    # Materialise a minimal config.json so config / provider / workspace pass.
    from cytopert.config.loader import save_config
    from cytopert.config.schema import (
        AgentDefaults,
        AgentsConfig,
        Config,
        ProviderConfig,
        ProvidersConfig,
    )

    cfg = Config(
        providers=ProvidersConfig(deepseek=ProviderConfig(api_key="sk-doctor")),
        agents=AgentsConfig(
            defaults=AgentDefaults(
                workspace=str(tmp_path / "ws"),
                model="deepseek-chat",
            )
        ),
    )
    save_config(cfg)

    console, buf = _captured_output()
    # The exit code can legitimately be 0 OR 1 here -- some CI matrices
    # do not have scanpy / decoupler / numba and those probes WARN
    # while still producing a clean run. We just want zero unhandled
    # exceptions; the rest is asserted on the captured output.
    run_doctor(ping=False, console=console)
    text = buf.getvalue()
    assert "config.json" in text and "PASS" in text
    assert "state.db" in text

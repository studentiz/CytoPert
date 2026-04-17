"""Stage 10.A regression tests for the cytopert setup wizard."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cytopert.cli import setup_wizard
from cytopert.config.loader import get_config_path, load_config


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def test_provider_models_table_complete() -> None:
    # Every known provider has at least one suggested model.
    for prov in ("openrouter", "deepseek", "anthropic", "openai", "vllm"):
        assert prov in setup_wizard.PROVIDER_MODELS
        assert setup_wizard.PROVIDER_MODELS[prov], prov


def test_build_config_round_trips() -> None:
    choices = setup_wizard.WizardChoices(
        provider="deepseek",
        api_key="sk-test",
        api_base=None,
        model="deepseek-chat",
        workspace=str(Path("~/wstest").expanduser()),
    )
    cfg = setup_wizard._build_config(choices)
    assert cfg.providers.deepseek.api_key == "sk-test"
    assert cfg.agents.defaults.model == "deepseek-chat"
    assert cfg.get_provider_type() == "deepseek"


def test_run_test_call_no_key_returns_warning(tmp_path: Path) -> None:
    choices = setup_wizard.WizardChoices(
        provider="deepseek",
        api_key="",
        api_base=None,
        model="deepseek-chat",
        workspace=str(tmp_path / "ws"),
    )
    cfg = setup_wizard._build_config(choices)
    ok, msg = asyncio.run(setup_wizard._run_test_call(cfg))
    assert ok is False
    assert "no api key" in msg.lower()


def test_run_wizard_writes_config(monkeypatch, tmp_path) -> None:
    """End-to-end: stub Prompt / Confirm and the test call; assert the
    resulting config.json is valid and load_config() agrees with it."""
    answers = iter([
        "deepseek",                         # provider pick
        "sk-test-end-to-end",               # API key
        "deepseek-chat",                    # model pick
        str(tmp_path / "wsx"),              # workspace
        "3",                                # test failed -> save anyway
    ])

    def _ask(*args, **kwargs):
        return next(answers)

    monkeypatch.setattr(setup_wizard.Prompt, "ask", _ask)
    # Confirm.ask is used both by the overwrite prompt and by the
    # state-reset prompt; default-False on the latter is fine here
    # because the tempdir has no prior state to wipe.
    monkeypatch.setattr(setup_wizard.Confirm, "ask", lambda *a, **k: False)
    # Skip the actual LLM round-trip; pretend it failed harmlessly.
    monkeypatch.setattr(
        setup_wizard, "_run_test_call",
        lambda cfg: asyncio.sleep(0, result=(False, "mocked")),
    )

    cfg = setup_wizard.run_wizard()
    assert get_config_path().exists()
    reloaded = load_config()
    assert reloaded.providers.deepseek.api_key == "sk-test-end-to-end"
    assert reloaded.agents.defaults.model == "deepseek-chat"
    assert cfg.get_provider_type() == "deepseek"


def test_run_wizard_retries_with_new_key_when_test_fails(monkeypatch, tmp_path) -> None:
    """First test_call fails, user picks 'retry with new key', second succeeds."""
    answers = iter([
        "deepseek",                  # provider
        "sk-bad",                    # initial API key
        "deepseek-chat",             # model
        str(tmp_path / "wsy"),       # workspace
        "1",                         # retry path: re-enter API key
        "sk-good",                   # new key
    ])

    def _ask(*args, **kwargs):
        return next(answers)

    call_count = {"n": 0}

    async def _fake_test(cfg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (False, "first attempt mocked-fail")
        return (True, "second attempt mocked-pass")

    monkeypatch.setattr(setup_wizard.Prompt, "ask", _ask)
    monkeypatch.setattr(setup_wizard.Confirm, "ask", lambda *a, **k: False)
    monkeypatch.setattr(setup_wizard, "_run_test_call", _fake_test)

    cfg = setup_wizard.run_wizard()
    assert call_count["n"] == 2
    # The reconfigured key is what landed in the saved config.
    reloaded = load_config()
    assert reloaded.providers.deepseek.api_key == "sk-good"
    assert cfg.providers.deepseek.api_key == "sk-good"


def test_maybe_clear_prior_state_skips_when_nothing_exists(monkeypatch, tmp_path) -> None:
    """No-op when the data dir has no memory / chains / sessions yet."""
    from rich.console import Console

    monkeypatch.setattr(setup_wizard.Confirm, "ask", lambda *a, **k: True)
    setup_wizard._maybe_clear_prior_state(Console())  # must not raise

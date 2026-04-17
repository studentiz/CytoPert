"""Stage 11 sanity tests for the prompt_toolkit shell scaffolding.

We do not run the full prompt_toolkit event loop here -- it requires a
TTY and pumping a pipe input is awkward to coordinate with rich's Live
panel inside pytest. Instead we exercise the pieces that do not need a
running shell:
    * the slash-word completer list,
    * the bottom-toolbar render,
    * importability (the ``cytopert agent`` fallback path relies on a
      successful import of cytopert.cli.interactive).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cytopert.agent.loop import AgentLoop
from cytopert.providers.base import LLMProvider, LLMResponse


class _NoopProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="x")

    async def chat(self, **_kwargs):
        return LLMResponse(content="ok", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def test_module_imports_without_optional_deps() -> None:
    # Importing the module must not require prompt_toolkit at import
    # time; it imports lazily inside run_prompt_toolkit_shell. This
    # test pins that contract so the CLI's fallback path keeps working.
    from cytopert.cli import interactive

    assert hasattr(interactive, "run_prompt_toolkit_shell")


def test_slash_words_includes_help_and_exit() -> None:
    from cytopert.cli.interactive import _slash_words

    words = _slash_words()
    assert "/help" in words
    assert "/exit" in words
    assert "/quit" in words
    assert "/model" in words


def test_render_toolbar_shows_model_and_plan(tmp_path: Path) -> None:
    from cytopert.cli.interactive import _render_toolbar

    loop = AgentLoop(
        provider=_NoopProvider(),
        workspace=tmp_path / "ws",
        model="stub-model",
        enable_reflection=False,
        load_plugins=False,
    )
    loop.enable_plan_gate("smoke")
    bar = _render_toolbar(loop, "smoke")
    assert "stub-model" in bar
    assert "plan=" in bar
    assert "calls=" in bar
    assert "cost=$0.00" in bar

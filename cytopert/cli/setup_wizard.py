"""Interactive first-time setup wizard for CytoPert.

Adapted in spirit from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:hermes_cli/setup.py (MIT
License). Hermes' wizard is ~3000 lines because it has to deal with
18 messaging platforms, OAuth flows, voice TTS, profile migration, and
an OpenClaw importer. CytoPert's needs are far simpler so this file is
deliberately ~250 lines: pick provider, enter key, pick model, pick
workspace, run a test call, install bundled skills, print the next
steps. See docs/hermes-borrowing.md for the full diff rationale.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from cytopert import __logo__
from cytopert.config.loader import get_config_path, load_config, save_config
from cytopert.config.schema import (
    AgentDefaults,
    AgentsConfig,
    Config,
    ProviderConfig,
    ProvidersConfig,
)

# Per-provider model suggestions. The first entry is the default.
PROVIDER_MODELS: dict[str, list[str]] = {
    "openrouter": [
        "anthropic/claude-sonnet-4-20250514",
        "openai/gpt-5",
        "google/gemini-2.5-pro",
        "deepseek/deepseek-chat-v3.1",
    ],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "claude-opus-4-5",
    ],
    "openai": ["gpt-5", "gpt-4o", "gpt-4o-mini"],
    "vllm": ["Qwen/Qwen3-30B-A3B-Instruct-2507"],
}

# Per-provider default API base URL (None means use the SDK default).
PROVIDER_DEFAULT_BASE: dict[str, str | None] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": None,
    "anthropic": None,
    "openai": None,
    "vllm": "http://localhost:8000/v1",
}

# Per-provider help URL for obtaining the API key.
PROVIDER_KEY_URL: dict[str, str] = {
    "openrouter": "https://openrouter.ai/keys",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
    "vllm": "(self-hosted; any non-empty string)",
}


@dataclass
class WizardChoices:
    """Captured user input from the wizard, ready to materialise as a Config."""

    provider: str
    api_key: str
    api_base: str | None
    model: str
    workspace: str


def _ascii_logo() -> str:
    return r"""
   ____      _       ____            _
  / ___|   _| |_ ___|  _ \ ___ _ __| |_
 | |  | | | | __/ _ \ |_) / _ \ '__| __|
 | |__| |_| | ||  __/  __/  __/ |  | |_
  \____\__, |\__\___|_|   \___|_|   \__|
       |___/    Single-cell research agent
"""


def _ask_provider(console: Console) -> str:
    table = Table(title="Available LLM providers", show_lines=False)
    table.add_column("Choice", style="bold")
    table.add_column("Provider")
    table.add_column("Get an API key")
    choices = list(PROVIDER_MODELS.keys())
    for i, prov in enumerate(choices, start=1):
        table.add_row(str(i), prov, PROVIDER_KEY_URL.get(prov, ""))
    console.print(table)
    raw = Prompt.ask(
        "Pick a provider by number or name",
        default=choices[0],
        show_default=True,
    )
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    if raw in choices:
        return raw
    console.print(f"[red]Unknown provider {raw!r}; defaulting to {choices[0]}[/red]")
    return choices[0]


def _ask_api_key(console: Console, provider: str) -> str:
    url = PROVIDER_KEY_URL.get(provider, "")
    if url:
        console.print(f"[dim]Get an {provider} API key from {url}[/dim]")
    while True:
        key = Prompt.ask(
            f"Paste your {provider} API key (or 'skip' to leave blank)",
            password=True,
        )
        if key.strip().lower() == "skip":
            return ""
        if key.strip():
            return key.strip()
        console.print("[yellow]Key was empty; try again or type 'skip'.[/yellow]")


def _ask_model(console: Console, provider: str) -> str:
    suggestions = PROVIDER_MODELS.get(provider, [])
    if not suggestions:
        return Prompt.ask(f"Model name for {provider}")
    table = Table(title=f"Suggested {provider} models", show_lines=False)
    table.add_column("Choice", style="bold")
    table.add_column("Model")
    for i, mdl in enumerate(suggestions, start=1):
        table.add_row(str(i), mdl)
    console.print(table)
    raw = Prompt.ask(
        "Pick a model by number, or paste any other model name",
        default=suggestions[0],
        show_default=True,
    )
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(suggestions):
            return suggestions[idx]
    return raw or suggestions[0]


def _ask_workspace(console: Console, default: str) -> str:
    raw = Prompt.ask(
        "Workspace directory for scanpy intermediate .h5ad outputs",
        default=default,
        show_default=True,
    )
    return raw or default


def _build_config(choices: WizardChoices) -> Config:
    providers = ProvidersConfig()
    setattr(
        providers,
        choices.provider,
        ProviderConfig(api_key=choices.api_key, api_base=choices.api_base),
    )
    return Config(
        providers=providers,
        agents=AgentsConfig(
            defaults=AgentDefaults(workspace=choices.workspace, model=choices.model)
        ),
    )


async def _run_test_call(config: Config) -> tuple[bool, str]:
    """Make a 1-token round-trip to confirm the credentials work.

    Returns ``(ok, message)``. ``ok=False`` when no API key is set or
    the call raised; we treat the latter as a soft warning, not a fatal
    error, because the user might be configuring an offline vLLM
    endpoint that is not yet running.
    """
    if not config.get_api_key():
        return (False, "No API key entered; skipped the test call.")
    try:
        from cytopert.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(
            api_key=config.get_api_key(),
            api_base=config.get_api_base(),
            default_model=config.agents.defaults.model,
            provider_type=config.get_provider_type(),
        )
        response = await provider.chat(
            messages=[
                {"role": "user", "content": "Reply with the single word 'ok'."}
            ],
            tools=None,
            model=config.agents.defaults.model,
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 -- wizard absorbs all failures
        return (False, f"Test call raised: {exc}")
    if response.finish_reason == "error":
        return (False, f"Provider returned error: {response.content}")
    text = (response.content or "").strip()
    return (True, f"Provider replied with {text[:40]!r}")


def _install_bundled_skills(console: Console) -> int:
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import get_skills_dir

    try:
        return SkillsManager(get_skills_dir()).install_bundled()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Could not install bundled skills: {exc}[/yellow]")
        return 0


def _print_summary(
    console: Console,
    config_path: Path,
    config: Config,
    test_ok: bool,
    test_msg: str,
    skills_installed: int,
) -> None:
    body_rows = [
        f"Config       : {config_path}",
        f"Provider     : {config.get_provider_type()}",
        f"Model        : {config.agents.defaults.model}",
        f"Workspace    : {config.workspace_path}",
        f"Test call    : {'OK' if test_ok else 'WARN'} - {test_msg}",
        f"Skills        : {skills_installed} bundled skill(s) installed",
    ]
    console.print(Panel("\n".join(body_rows), title="CytoPert setup complete"))
    console.print(
        "\nNext steps:\n"
        "  1. cytopert agent                # interactive shell (PlanGate enabled by default)\n"
        "  2. cytopert agent -m \"hello\"     # one-shot prompt\n"
        "  3. cytopert run-workflow generic_de --question \"...\"   # template scenario\n"
        "  4. cytopert doctor               # health-check the install\n"
    )


def run_wizard(
    *,
    console: Console | None = None,
    pre_picked_provider: str | None = None,
) -> Config:
    """Run the interactive setup wizard end-to-end and return the saved Config.

    ``pre_picked_provider`` is for tests; in normal use the wizard
    prompts the user. The function never raises on test-call failure --
    it logs a warning and still saves the config so the user can retry
    the call with ``cytopert doctor --ping`` after editing the key.
    """
    console = console or Console()
    console.print(_ascii_logo(), style="bold")
    console.print(f"{__logo__} CytoPert first-time setup\n")

    config_path = get_config_path()
    if config_path.exists():
        if not Confirm.ask(
            f"Config already exists at {config_path}. Overwrite?", default=False
        ):
            console.print("[yellow]Setup aborted; existing config left intact.[/yellow]")
            return load_config()

    provider = pre_picked_provider or _ask_provider(console)
    api_key = _ask_api_key(console, provider)
    model = _ask_model(console, provider)
    workspace_default = str(Path("~/.cytopert/workspace").expanduser())
    workspace = _ask_workspace(console, workspace_default)
    api_base = PROVIDER_DEFAULT_BASE.get(provider)

    choices = WizardChoices(
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        model=model,
        workspace=workspace,
    )
    config = _build_config(choices)
    save_config(config)

    test_ok, test_msg = asyncio.run(_run_test_call(config))
    skills_installed = _install_bundled_skills(console)
    _print_summary(console, config_path, config, test_ok, test_msg, skills_installed)
    return config


def setup_command_callback() -> None:
    """Typer entry point for ``cytopert setup``."""
    try:
        run_wizard()
    except (KeyboardInterrupt, EOFError):
        Console().print("\n[yellow]Setup cancelled.[/yellow]")
        raise typer.Exit(130)


__all__ = [
    "PROVIDER_MODELS",
    "PROVIDER_KEY_URL",
    "WizardChoices",
    "run_wizard",
    "setup_command_callback",
]

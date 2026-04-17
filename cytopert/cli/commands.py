"""CLI commands for CytoPert."""

import asyncio
import json
import os

import typer
from rich.console import Console
from rich.table import Table

from cytopert import __logo__, __version__
from cytopert.cli.profiles import profile_app

app = typer.Typer(
    name="cytopert",
    help=f"{__logo__} CytoPert - Cell perturbation differential response mechanism parsing",
    no_args_is_help=True,
)

memory_app = typer.Typer(name="memory", help="Manage CytoPert persistent memory.", no_args_is_help=True)
skills_app = typer.Typer(name="skills", help="Manage CytoPert procedural skills.", no_args_is_help=True)
chains_app = typer.Typer(name="chains", help="Inspect mechanism chain lifecycle.", no_args_is_help=True)
evidence_app = typer.Typer(name="evidence", help="Search persistent evidence (cross-session).", no_args_is_help=True)
plugins_app = typer.Typer(
    name="plugins",
    help="List / enable / disable CytoPert plugins (user, project, entry-points).",
    no_args_is_help=True,
)
config_app = typer.Typer(
    name="config",
    help="Quick get/set on individual config keys (no JSON editing required).",
    no_args_is_help=True,
)
cron_app = typer.Typer(
    name="cron",
    help="Schedule recurring agent / workflow runs (jobs.json + tick / daemon).",
    no_args_is_help=True,
)

app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
app.add_typer(chains_app, name="chains")
app.add_typer(evidence_app, name="evidence")
app.add_typer(plugins_app, name="plugins")
app.add_typer(config_app, name="config")
app.add_typer(profile_app, name="profile")
app.add_typer(cron_app, name="cron")

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"{__logo__} cytopert v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(None, "--version", "-v", callback=_version_callback, is_eager=True),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help=(
            "Switch to a named profile for THIS process only. Equivalent "
            "to setting CYTOPERT_HOME=~/.cytopert/profiles/<name>. The "
            "directory is created on demand. Persist the choice across "
            "invocations with `cytopert profile use <name>`."
        ),
    ),
) -> None:
    """CytoPert - Interactive framework for single-cell perturbation analysis."""
    if profile:
        # Look the constants up via the module object so test
        # monkeypatching of ``cytopert.utils.helpers.CYTOPERT_ROOT_DIR``
        # propagates here. ``from helpers import X`` would freeze X to
        # the original value at the time of the ``from`` statement.
        from cytopert.utils import helpers as hh

        target = hh.CYTOPERT_ROOT_DIR / hh.PROFILES_SUBDIR / profile
        hh.ensure_dir(target)
        os.environ["CYTOPERT_HOME"] = str(target)


@app.command()
def setup() -> None:
    """Interactive first-time setup wizard (provider / key / model / workspace)."""
    from cytopert.cli.setup_wizard import setup_command_callback

    setup_command_callback()


@app.command()
def onboard() -> None:
    """Initialize CytoPert configuration, workspace, memory, and skills directories."""
    from cytopert.config.loader import get_config_path, save_config
    from cytopert.config.schema import Config
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import (
        get_chains_dir,
        get_memory_dir,
        get_skills_dir,
        get_workspace_path,
    )

    config_path = get_config_path()
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit()
    config = Config()
    save_config(config)
    console.print(f"[green]✓[/green] Created config at {config_path}")
    workspace = get_workspace_path()
    console.print(f"[green]✓[/green] Created workspace at {workspace}")
    memory_dir = get_memory_dir()
    console.print(f"[green]✓[/green] Memory dir at {memory_dir}")
    chains_dir = get_chains_dir()
    console.print(f"[green]✓[/green] Chains dir at {chains_dir}")
    skills_dir = get_skills_dir()
    n = SkillsManager(skills_dir).install_bundled()
    console.print(f"[green]✓[/green] Skills dir at {skills_dir} (bundled installed: {n})")
    console.print(f"\n{__logo__} CytoPert is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.cytopert/config.json[/cyan]")
    console.print("     (e.g. providers.openrouter.apiKey from https://openrouter.ai/keys)")
    console.print("  2. Chat: [cyan]cytopert agent -m \"Your question\"[/cyan]")
    console.print(
        "  3. Run a bundled example workflow: "
        "[cyan]cytopert run-workflow <scenario>[/cyan] "
        "(see [cyan]cytopert run-workflow --help[/cyan] for the registered names)."
    )
    console.print("  4. Inspect learnings: [cyan]cytopert memory show / cytopert skills list[/cyan]")


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:default", "--session", "-s", help="Session ID"),
    feedback: str | None = typer.Option(
        None,
        "--feedback",
        "-f",
        help=(
            "Wet-lab or experiment feedback for this turn. Forwarded verbatim to the "
            "reflection module so it can advance chain status / update memory."
        ),
    ),
    save_trajectory: bool = typer.Option(
        False,
        "--save-trajectory",
        help=(
            "Append a ShareGPT-format trajectory entry to "
            "~/.cytopert/trajectories/trajectory_samples.jsonl after each turn. "
            "Off by default to avoid surprising disk writes."
        ),
    ),
) -> None:
    """Interact with the agent (interactive or single message)."""
    from cytopert.agent.loop import AgentLoop
    from cytopert.config.loader import load_config
    from cytopert.providers.litellm_provider import LiteLLMProvider

    config = load_config()
    api_key = config.get_api_key()
    api_base = config.get_api_base()
    model = config.agents.defaults.model
    if not api_key and not (api_base and "vllm" in (api_base or "").lower()):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Run [cyan]cytopert onboard[/cyan] and set providers in ~/.cytopert/config.json")
        raise typer.Exit(1)
    provider = LiteLLMProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model,
        provider_type=config.get_provider_type(),
    )
    agent_loop = AgentLoop(
        provider=provider,
        workspace=config.workspace_path,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        # Thread the configured generation defaults so config.json values
        # for temperature / max_tokens actually reach provider.chat instead
        # of silently falling back to LiteLLM's library defaults.
        max_tokens=config.agents.defaults.max_tokens,
        temperature=config.agents.defaults.temperature,
        save_trajectory=save_trajectory,
    )
    if message:
        async def run_once() -> None:
            response = await agent_loop.process_direct(
                message, session_id, user_feedback=feedback
            )
            console.print(f"\n{__logo__} {response}")

        asyncio.run(run_once())
    else:
        # Interactive shell: prompt_toolkit when available (full TUI:
        # multiline, slash autocomplete, history, streaming, status
        # line, Ctrl+C cancel-then-exit). Falls back to the basic
        # rich.console loop if prompt_toolkit is missing for any
        # reason (allows the agent to run in dependency-stripped CI).
        from cytopert.cli.interactive_slash import handle_slash_command

        try:
            from cytopert.cli.interactive import run_prompt_toolkit_shell

            asyncio.run(run_prompt_toolkit_shell(agent_loop, session_id, feedback))
            return
        except ImportError as exc:
            console.print(
                f"[yellow]prompt_toolkit unavailable ({exc}); "
                "falling back to the basic shell.[/yellow]"
            )

        console.print(f"{__logo__} Interactive mode (Ctrl+C to exit)\n")
        console.print(
            "[dim]Type /help for the full slash-command list. Plan-gate is "
            "ON by default; reply 'go' (or 'execute' / 'approve') to "
            "authorise tool calls in the next turn.[/dim]\n"
        )
        # Interactive sessions start in plan-then-execute mode: the first
        # turn produces a textual plan only and tool calls are gated until
        # the researcher types 'go'. Use /skip-plan to disable it for
        # casual chitchat sessions.
        agent_loop.enable_plan_gate(session_id)

        async def run_interactive() -> None:
            nonlocal feedback
            while True:
                try:
                    user_input = console.input("[bold blue]You:[/bold blue] ")
                    if not user_input.strip():
                        continue
                    if user_input.lstrip().startswith("/"):
                        verdict = handle_slash_command(
                            user_input, agent_loop, session_id, console
                        )
                        if verdict == "exit":
                            break
                        if verdict == "handled":
                            continue
                        # verdict == "passthrough": fall through to LLM
                    # --feedback only applies to the first turn (matching
                    # the workflow scenario semantics); subsequent
                    # interactive turns leave it empty.
                    response = await agent_loop.process_direct(
                        user_input, session_id, user_feedback=feedback
                    )
                    feedback = None
                    console.print(f"\n{__logo__} {response}\n")
                except KeyboardInterrupt:
                    console.print("\nGoodbye!")
                    break

        asyncio.run(run_interactive())


@app.command()
def run_workflow(
    scenario: str = typer.Argument(..., help="Scenario name (e.g. nfatc1_mammary)"),
    feedback: str | None = typer.Option(None, "--feedback", "-f", help="Experiment feedback for next round"),
    question: str | None = typer.Option(None, "--question", "-q", help="Research question (otherwise prompt)"),
) -> None:
    """Run a workflow scenario via the SCENARIO_REGISTRY."""
    from cytopert.agent.loop import AgentLoop
    from cytopert.config.loader import load_config
    from cytopert.providers.litellm_provider import LiteLLMProvider
    from cytopert.workflow.pipeline import (
        StageContext,
        available_scenarios,
        get_scenario,
        get_scenario_config,
    )

    config = load_config()
    if not config.get_api_key():
        console.print("[red]Error: No API key configured.[/red]")
        raise typer.Exit(1)

    # Importing the scenarios package triggers autoimport of every
    # bundled module so register_scenario calls run before lookup.
    pipeline = get_scenario(scenario)
    if pipeline is None:
        known = ", ".join(available_scenarios()) or "(none)"
        console.print(
            f"[yellow]Unknown scenario: {scenario}. Registered: {known}.[/yellow]"
        )
        console.print(
            "[dim]Add a module under cytopert/workflow/scenarios/ "
            "and call register_scenario(name, factory).[/dim]"
        )
        raise typer.Exit(2)

    research_question = question or (
        f"Identify differential response mechanism for scenario {scenario}."
    )
    provider = LiteLLMProvider(
        api_key=config.get_api_key(),
        api_base=config.get_api_base(),
        default_model=config.agents.defaults.model,
        provider_type=config.get_provider_type(),
    )
    agent = AgentLoop(
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        max_tokens=config.agents.defaults.max_tokens,
        temperature=config.agents.defaults.temperature,
    )
    ctx = StageContext(
        config=config,
        research_question=research_question,
        data_config=get_scenario_config(config, scenario) or {},
        feedback=feedback,
        session_key=f"workflow:{scenario}",
    )
    result = asyncio.run(pipeline.run(agent, ctx))
    console.print(
        f"\n{__logo__} [bold]Workflow result ({pipeline.name}):[/bold]\n"
    )
    console.print(result["response"])
    if feedback:
        console.print(f"\n[dim]Feedback applied: {feedback}[/dim]")


@app.command()
def status() -> None:
    """Show CytoPert status."""
    from cytopert.config.loader import get_config_path, load_config
    from cytopert.memory.store import MemoryStore
    from cytopert.persistence.chain_db import ChainStore
    from cytopert.persistence.evidence_db import EvidenceDB
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import (
        active_profile_name,
        get_chains_dir,
        get_memory_dir,
        get_skills_dir,
        get_state_db_path,
    )

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path
    console.print(f"{__logo__} CytoPert Status\n")
    profile = active_profile_name()
    if profile:
        console.print(f"Profile: [bold]{profile}[/bold]")
    else:
        console.print("Profile: [dim]default root[/dim]")
    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")
    if config_path.exists():
        console.print(f"Model: {config.agents.defaults.model}")
        provider_type = config.get_provider_type() or "[dim]not set[/dim]"
        console.print(f"Provider: {provider_type}")
        has_key = bool(config.get_api_key())
        console.print(f"API key: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")

    db = EvidenceDB(get_state_db_path())
    chains = ChainStore(get_state_db_path(), get_chains_dir())
    skills = SkillsManager(get_skills_dir())
    memory = MemoryStore(get_memory_dir())
    console.print()
    console.print(f"Evidence entries (cross-session): [bold]{db.count()}[/bold]")
    console.print(
        "Chains: "
        f"proposed={chains.count('proposed')}  supported={chains.count('supported')}  "
        f"refuted={chains.count('refuted')}  superseded={chains.count('superseded')}"
    )
    console.print(f"Skills installed: [bold]{len(skills.list())}[/bold] "
                  f"(staged: {len(skills.list(include_staged=True)) - len(skills.list())})")
    for target in ("context", "researcher", "hypothesis_log"):
        chars, limit = memory.usage(target)
        pct = int(round(100.0 * chars / limit)) if limit else 0
        console.print(rf"Memory\[{target}]: {chars}/{limit} chars ({pct}%)")


@app.command()
def doctor(
    ping: bool = typer.Option(
        False,
        "--ping",
        help="Also issue a 1-token LLM round-trip; off by default to save tokens.",
    ),
) -> None:
    """Run end-to-end health checks on the CytoPert install."""
    from cytopert.cli.doctor import run_doctor

    exit_code = run_doctor(ping=ping)
    raise typer.Exit(exit_code)


@app.command()
def model(
    name: str = typer.Argument(
        None,
        help="New default model. Omit to print the current model + suggestions.",
    ),
) -> None:
    """Show or persist the default model used by ``cytopert agent``."""
    from cytopert.cli.setup_wizard import PROVIDER_MODELS
    from cytopert.config.loader import load_config, save_config

    cfg = load_config()
    if name is None:
        console.print(f"Current model: [bold]{cfg.agents.defaults.model}[/bold]")
        provider = cfg.get_provider_type() or "openrouter"
        suggestions = PROVIDER_MODELS.get(provider, [])
        if suggestions:
            console.print(f"\nSuggested models for provider [cyan]{provider}[/cyan]:")
            for s in suggestions:
                marker = " (current)" if s == cfg.agents.defaults.model else ""
                console.print(f"  - {s}{marker}")
        return
    cfg.agents.defaults.model = name
    save_config(cfg)
    console.print(f"[green]\u2713[/green] Default model set to [bold]{name}[/bold].")


def _split_dotted(path: str) -> list[str]:
    """Split a dotted config path; rejects empty segments."""
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise typer.BadParameter("Path must be non-empty (e.g. agents.defaults.model)")
    return parts


def _get_dotted(data: object, parts: list[str]) -> object:
    cur: object = data
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(".".join(parts))
        cur = cur[p]
    return cur


def _set_dotted(data: dict, parts: list[str], value: object) -> None:
    cur: dict = data
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            cur[p] = {}
            nxt = cur[p]
        cur = nxt
    cur[parts[-1]] = value


def _coerce_value(raw: str) -> object:
    """Parse a CLI-supplied value as JSON when possible; fall back to str."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


@config_app.command("get")
def config_get(path: str = typer.Argument(..., help="Dotted config path")) -> None:
    """Print the current value at ``path`` (e.g. agents.defaults.model)."""
    from cytopert.config.loader import get_config_path
    parts = _split_dotted(path)
    cfg_path = get_config_path()
    if not cfg_path.exists():
        console.print(f"[red]No config at {cfg_path}; run cytopert setup first.[/red]")
        raise typer.Exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        value = _get_dotted(data, parts)
    except KeyError:
        console.print(f"[red]No such key: {path}[/red]")
        raise typer.Exit(2) from None
    if isinstance(value, (dict, list)):
        console.print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        console.print(value)


@config_app.command("set")
def config_set(
    path: str = typer.Argument(..., help="Dotted config path"),
    value: str = typer.Argument(..., help="New value (parsed as JSON when possible)"),
) -> None:
    """Set the value at ``path`` and re-validate the resulting config."""
    from cytopert.config.loader import get_config_path
    from cytopert.config.schema import Config

    parts = _split_dotted(path)
    cfg_path = get_config_path()
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
    parsed = _coerce_value(value)
    _set_dotted(data, parts, parsed)
    # Re-validate by round-tripping through the Pydantic schema.
    from cytopert.config.loader import _convert_keys, _convert_to_camel
    try:
        Config.model_validate(_convert_keys(data))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Resulting config is invalid: {exc}[/red]")
        raise typer.Exit(2) from None
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(_convert_to_camel(data), f, indent=2)
    console.print(
        f"[green]\u2713[/green] Set [bold]{path}[/bold] = "
        f"{json.dumps(parsed, ensure_ascii=False)}"
    )


# ---------------------------------------------------------------------------
# memory subcommands
# ---------------------------------------------------------------------------

_MEMORY_TARGETS = ("context", "researcher", "hypothesis_log")


def _memory_store():
    from cytopert.memory.store import MemoryStore
    from cytopert.utils.helpers import get_memory_dir

    return MemoryStore(get_memory_dir())


@memory_app.command("show")
def memory_show(
    target: str = typer.Option(
        None, "--target", "-t",
        help=f"Show only this target ({', '.join(_MEMORY_TARGETS)}). Omit to show all.",
    ),
) -> None:
    """Print the current snapshot for one or all memory targets."""
    store = _memory_store()
    if target:
        if target not in _MEMORY_TARGETS:
            console.print(f"[red]Unknown target {target!r}; expected one of {_MEMORY_TARGETS}[/red]")
            raise typer.Exit(2)
        chars, limit = store.usage(target)
        console.print(f"[bold]{target}[/bold] [{chars}/{limit} chars]")
        body = store.read(target).strip() or "(empty)"
        console.print(body)
        return
    console.print(store.render_snapshot())


@memory_app.command("edit")
def memory_edit(
    target: str = typer.Argument(..., help="context | researcher | hypothesis_log"),
) -> None:
    """Open the chosen memory file in $EDITOR (or VISUAL)."""
    if target not in _MEMORY_TARGETS:
        console.print(f"[red]Unknown target {target!r}[/red]")
        raise typer.Exit(2)
    store = _memory_store()
    path = store.memory_dir / {
        "context": "CONTEXT.md",
        "researcher": "RESEARCHER.md",
        "hypothesis_log": "HYPOTHESIS_LOG.md",
    }[target]
    path.touch(exist_ok=True)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    os.system(f'{editor} "{path}"')
    chars, limit = store.usage(target)
    if chars > limit:
        console.print(f"[yellow]Warning:[/yellow] {target} is now {chars}/{limit} chars (over limit).")
    else:
        console.print(f"[green]✓[/green] {target} saved ({chars}/{limit} chars).")


@memory_app.command("clear")
def memory_clear(
    target: str = typer.Option(None, "--target", "-t", help="Specific target; omit to clear all."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete memory file(s)."""
    store = _memory_store()
    if target and target not in _MEMORY_TARGETS:
        console.print(f"[red]Unknown target {target!r}[/red]")
        raise typer.Exit(2)
    if not yes and not typer.confirm(f"Clear memory{' [' + target + ']' if target else ''}?"):
        raise typer.Exit()
    store.clear(target)
    console.print("[green]✓[/green] Cleared.")


# ---------------------------------------------------------------------------
# skills subcommands
# ---------------------------------------------------------------------------

def _skills_manager():
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import get_skills_dir

    return SkillsManager(get_skills_dir())


@skills_app.command("list")
def skills_list(
    include_staged: bool = typer.Option(False, "--include-staged", "-S",
                                        help="Also list staged (auto-proposed, not-yet-accepted) skills."),
) -> None:
    """List installed skills (Level 0 metadata)."""
    mgr = _skills_manager()
    skills = mgr.list(include_staged=include_staged)
    if not skills:
        console.print("[dim]No skills installed.[/dim]")
        return
    table = Table(show_lines=False)
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("Staged")
    for s in skills:
        table.add_row(s.name, s.category, s.version, s.description,
                      "[yellow]yes[/yellow]" if s.staged else "")
    console.print(table)


@skills_app.command("show")
def skills_show(name: str = typer.Argument(...)) -> None:
    """Print full SKILL.md content."""
    mgr = _skills_manager()
    try:
        console.print(mgr.view(name))
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@skills_app.command("new")
def skills_new(
    name: str = typer.Argument(...),
    category: str = typer.Option("uncategorized", "--category", "-c"),
    description: str = typer.Option("", "--description", "-d"),
    staged: bool = typer.Option(False, "--staged/--live", help="Create as staged (default: live)."),
) -> None:
    """Scaffold a new SKILL.md and open it in $EDITOR."""
    mgr = _skills_manager()
    template = (
        f"---\nname: {name}\ndescription: {description or '(provide a description)'}\n"
        f"version: 0.1.0\nmetadata:\n  cytopert:\n    category: {category}\n    tags: []\n---\n\n"
        f"# {name}\n\n## When to Use\n\n## Procedure\n\n## Pitfalls\n\n## Verification\n"
    )
    try:
        path = mgr.create(name, template, category=category, staged=staged)
    except (FileExistsError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        os.system(f'{editor} "{path}"')
    console.print(f"[green]✓[/green] Created skill at {path}")


@skills_app.command("accept")
def skills_accept(
    name: str = typer.Argument(...),
    category: str = typer.Option(None, "--category", "-c"),
) -> None:
    """Promote a staged skill to the live skills directory."""
    mgr = _skills_manager()
    try:
        path = mgr.accept_staged(name, category=category)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Promoted {name} to {path}")


@skills_app.command("delete")
def skills_delete(name: str = typer.Argument(...),
                  yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete a skill (live or staged)."""
    if not yes and not typer.confirm(f"Delete skill {name!r}?"):
        raise typer.Exit()
    mgr = _skills_manager()
    try:
        mgr.delete(name)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Deleted {name}")


# ---------------------------------------------------------------------------
# chains subcommands
# ---------------------------------------------------------------------------

def _chain_store():
    from cytopert.persistence.chain_db import ChainStore
    from cytopert.utils.helpers import get_chains_dir, get_state_db_path

    return ChainStore(get_state_db_path(), get_chains_dir())


@chains_app.command("list")
def chains_list(
    status: str | None = typer.Option(None, "--status",
                                      help="Filter by lifecycle status."),
    gene: str | None = typer.Option(None, "--gene", help="Filter by gene substring."),
    limit: int = typer.Option(50, "--limit", "-n"),
) -> None:
    """List mechanism chains, most recently updated first."""
    store = _chain_store()
    rows = store.list(status=status, gene=gene, limit=limit)
    if not rows:
        console.print("[dim]No chains.[/dim]")
        return
    table = Table()
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Priority")
    table.add_column("Evidence")
    table.add_column("Summary")
    for chain, st in rows:
        table.add_row(chain.id, st, chain.priority,
                      ", ".join(chain.evidence_ids[:3]) + ("..." if len(chain.evidence_ids) > 3 else ""),
                      chain.summary[:80])
    console.print(table)


@chains_app.command("show")
def chains_show(chain_id: str = typer.Argument(...)) -> None:
    """Show chain details + lifecycle events."""
    store = _chain_store()
    chain = store.get(chain_id)
    if chain is None:
        console.print(f"[red]Chain {chain_id!r} not found.[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]{chain.id}[/bold] [{store.get_status(chain.id)}] priority={chain.priority}")
    console.print(f"Summary: {chain.summary}")
    console.print(f"Evidence IDs: {chain.evidence_ids}")
    if chain.links:
        console.print("Links:")
        for link in chain.links:
            console.print(f"  - {link.from_node} --[{link.relation}]--> {link.to_node}  "
                          f"evidence={link.evidence_ids}")
    console.print("\nEvents:")
    for ev in store.events(chain_id):
        console.print(f"  - [{ev['created_at']}] {ev['event_type']} status={ev['status']} "
                      f"note={ev['note']!r}")


# ---------------------------------------------------------------------------
# evidence subcommands
# ---------------------------------------------------------------------------

def _evidence_db():
    from cytopert.persistence.evidence_db import EvidenceDB
    from cytopert.utils.helpers import get_state_db_path

    return EvidenceDB(get_state_db_path())


@evidence_app.command("search")
def evidence_search(
    query: str = typer.Argument("", help="Free-text query (FTS5)"),
    gene: str | None = typer.Option(None, "--gene"),
    pathway: str | None = typer.Option(None, "--pathway"),
    tissue: str | None = typer.Option(None, "--tissue"),
    tool: str | None = typer.Option(None, "--tool", help="Exact tool name (e.g. scanpy_de)"),
    top_k: int = typer.Option(20, "--top-k", "-n"),
) -> None:
    """Search persistent evidence (cross-session, FTS5 + filters)."""
    db = _evidence_db()
    entries = db.search(query=query or None, gene=gene, pathway=pathway,
                        tissue=tissue, tool_name=tool, top_k=top_k)
    if not entries:
        console.print("[dim]No matching evidence.[/dim]")
        return
    table = Table()
    table.add_column("ID")
    table.add_column("Tool")
    table.add_column("Genes")
    table.add_column("Pathways")
    table.add_column("Summary")
    for e in entries:
        table.add_row(
            e.id,
            e.tool_name or "",
            ", ".join(e.genes[:6]) + ("..." if len(e.genes) > 6 else ""),
            ", ".join(e.pathways[:6]) + ("..." if len(e.pathways) > 6 else ""),
            (e.summary or "")[:100],
        )
    console.print(table)


@evidence_app.command("show")
def evidence_show(evidence_id: str = typer.Argument(...)) -> None:
    """Print one evidence entry as JSON."""
    db = _evidence_db()
    entry = db.get(evidence_id)
    if entry is None:
        console.print(f"[red]No evidence with id {evidence_id!r}[/red]")
        raise typer.Exit(1)
    console.print(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, indent=2))


@evidence_app.command("recent")
def evidence_recent(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """Show the most recent evidence entries."""
    db = _evidence_db()
    entries = db.recent(limit=limit)
    if not entries:
        console.print("[dim]No evidence yet.[/dim]")
        return
    for e in entries:
        console.print(f"[bold]{e.id}[/bold] tool={e.tool_name} genes={e.genes[:5]}")
        console.print(f"  {e.summary[:120]}")


# ---------------------------------------------------------------------------
# plugins subcommands (stage 7.3)
# ---------------------------------------------------------------------------

def _plugin_manager():
    from pathlib import Path

    from cytopert.plugins.manager import PluginManager

    return PluginManager(project_dir=Path.cwd())


@plugins_app.command("list")
def plugins_list() -> None:
    """List discovered CytoPert plugins (user / project / entry-points)."""
    mgr = _plugin_manager()
    infos = mgr.discover()
    if not infos:
        console.print("[dim]No plugins discovered.[/dim]")
        return
    table = Table(show_lines=False)
    table.add_column("Name")
    table.add_column("Source")
    table.add_column("Enabled")
    table.add_column("Location")
    for info in infos:
        enabled = "[green]\u2713[/green]" if info.enabled else "[red]disabled[/red]"
        table.add_row(info.name, info.source.value, enabled, info.location)
    console.print(table)


def _disabled_path():
    from pathlib import Path

    from cytopert.plugins.manager import DEFAULT_DISABLED_FILE
    from cytopert.utils.helpers import get_data_path

    return Path(get_data_path()) / "plugins" / DEFAULT_DISABLED_FILE


def _toggle_disabled(name: str, *, disable: bool) -> None:
    path = _disabled_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.exists():
        existing = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    if disable:
        existing.add(name)
        action = "disabled"
    else:
        existing.discard(name)
        action = "enabled"
    path.write_text("\n".join(sorted(existing)) + ("\n" if existing else ""), encoding="utf-8")
    console.print(f"[green]\u2713[/green] {action} plugin {name!r}.")


@plugins_app.command("disable")
def plugins_disable(name: str = typer.Argument(...)) -> None:
    """Disable a plugin by name (writes ~/.cytopert/plugins/disabled.txt)."""
    _toggle_disabled(name, disable=True)


@plugins_app.command("enable")
def plugins_enable(name: str = typer.Argument(...)) -> None:
    """Re-enable a previously disabled plugin."""
    _toggle_disabled(name, disable=False)


# ---------------------------------------------------------------------------
# cron subcommands (stage 13a)
# ---------------------------------------------------------------------------


def _job_store():
    from cytopert.scheduler.cron import JobStore, get_default_jobs_path

    return JobStore(get_default_jobs_path())


def _build_agent_loop_for_cron():
    """Construct an AgentLoop bound to the active config + provider."""
    from cytopert.agent.loop import AgentLoop
    from cytopert.config.loader import load_config
    from cytopert.providers.litellm_provider import LiteLLMProvider

    cfg = load_config()
    provider = LiteLLMProvider(
        api_key=cfg.get_api_key(),
        api_base=cfg.get_api_base(),
        default_model=cfg.agents.defaults.model,
        provider_type=cfg.get_provider_type(),
    )
    loop = AgentLoop(
        provider=provider,
        workspace=cfg.workspace_path,
        model=cfg.agents.defaults.model,
        max_iterations=cfg.agents.defaults.max_tool_iterations,
        max_tokens=cfg.agents.defaults.max_tokens,
        temperature=cfg.agents.defaults.temperature,
    )
    return loop, cfg


@cron_app.command("add")
def cron_add(
    schedule: str = typer.Argument(
        ..., help="Schedule, e.g. 'every 30m', 'every 6h', 'hourly', 'daily'"
    ),
    message: str | None = typer.Option(
        None, "--message", "-m", help="Free-text user message to send to the agent."
    ),
    scenario: str | None = typer.Option(
        None, "--scenario", "-s", help="Workflow scenario name to run instead of a message."
    ),
    feedback: str | None = typer.Option(
        None, "--feedback", "-f", help="Optional feedback string forwarded to reflection."
    ),
    job_id: str | None = typer.Option(
        None, "--id", help="Stable job id (defaults to a random hex token)."
    ),
) -> None:
    """Register a recurring job in the active profile's jobs.json."""
    from cytopert.scheduler.cron import Job, parse_schedule

    if (message is None) == (scenario is None):
        console.print("[red]Provide exactly one of --message / --scenario[/red]")
        raise typer.Exit(2)
    try:
        parse_schedule(schedule)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    job = Job.make(
        schedule=schedule,
        message=message,
        scenario=scenario,
        feedback=feedback,
        job_id=job_id,
    )
    store = _job_store()
    try:
        store.add(job)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    target = scenario or (message or "")
    console.print(
        f"[green]\u2713[/green] Added job [bold]{job.id}[/bold] "
        f"({schedule} -> {target[:60]}); next run [cyan]{job.next_run}[/cyan]"
    )


@cron_app.command("list")
def cron_list() -> None:
    """List all jobs in jobs.json with their next-run timestamps."""
    from cytopert.scheduler.cron import synchronous_runner_for_message

    store = _job_store()
    jobs = store.load()
    if not jobs:
        console.print("[dim]No jobs scheduled. Add one with `cytopert cron add`.[/dim]")
        return
    table = Table(title="CytoPert cron", show_lines=False)
    table.add_column("ID", style="bold")
    table.add_column("Enabled")
    table.add_column("Schedule")
    table.add_column("Target")
    table.add_column("Next run")
    table.add_column("Last run")
    table.add_column("Last status")
    for j in jobs:
        table.add_row(
            j.id,
            "[green]\u2713[/green]" if j.enabled else "[red]off[/red]",
            j.schedule,
            synchronous_runner_for_message(j.message or "", scenario=j.scenario),
            j.next_run or "-",
            j.last_run or "-",
            j.last_status or "-",
        )
    console.print(table)


@cron_app.command("remove")
def cron_remove(job_id: str = typer.Argument(...)) -> None:
    """Delete a job by id."""
    store = _job_store()
    if not store.remove(job_id):
        console.print(f"[red]No such job: {job_id}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]\u2713[/green] Removed {job_id}.")


@cron_app.command("enable")
def cron_enable(job_id: str = typer.Argument(...)) -> None:
    """Re-enable a previously disabled job."""
    store = _job_store()
    try:
        store.set_enabled(job_id, True)
    except KeyError:
        console.print(f"[red]No such job: {job_id}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]\u2713[/green] Enabled {job_id}.")


@cron_app.command("disable")
def cron_disable(job_id: str = typer.Argument(...)) -> None:
    """Disable a job (it will remain in jobs.json but not be picked up)."""
    store = _job_store()
    try:
        store.set_enabled(job_id, False)
    except KeyError:
        console.print(f"[red]No such job: {job_id}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]\u2713[/green] Disabled {job_id}.")


@cron_app.command("tick")
def cron_tick(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print due jobs but do not run them."
    ),
) -> None:
    """Run every due job exactly once and exit (use from system cron)."""
    from cytopert.scheduler.cron import _utcnow, make_agent_runner, run_due_jobs

    store = _job_store()
    if dry_run:
        now = _utcnow()
        due = [j for j in store.load() if j.is_due(now=now)]
        if not due:
            console.print("[dim]No due jobs.[/dim]")
            return
        console.print(f"Would run {len(due)} job(s):")
        for j in due:
            console.print(f"  - {j.id} ({j.schedule}) next_run={j.next_run}")
        return
    loop, cfg = _build_agent_loop_for_cron()
    runner = make_agent_runner(loop, config=cfg)
    ran = asyncio.run(run_due_jobs(store, runner, on_progress=console.print))
    if not ran:
        console.print("[dim]No due jobs.[/dim]")
        return
    for j in ran:
        colour = "green" if j.last_status == "ok" else "red"
        console.print(
            f"  [{colour}]{j.last_status}[/{colour}] {j.id} "
            f"-> next {j.next_run} (err={j.last_error or '-'})"
        )


@cron_app.command("daemon")
def cron_daemon(
    interval: int = typer.Option(60, "--interval", "-i", help="Tick interval in seconds."),
) -> None:
    """Run the scheduler in a sleep loop until Ctrl+C."""
    import asyncio

    from cytopert.scheduler.cron import make_agent_runner, run_daemon

    if interval < 5:
        console.print("[red]--interval must be >= 5 seconds[/red]")
        raise typer.Exit(2)
    store = _job_store()
    loop, cfg = _build_agent_loop_for_cron()
    runner = make_agent_runner(loop, config=cfg)
    console.print(
        f"{__logo__} cron daemon starting (interval={interval}s; Ctrl+C to stop)"
    )
    stop_event = asyncio.Event()

    async def _run() -> None:
        await run_daemon(
            store,
            runner,
            interval_seconds=interval,
            stop_event=stop_event,
            on_tick=lambda jobs: (
                console.print(f"  ran {len(jobs)} job(s)") if jobs else None
            ),
        )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        stop_event.set()
        console.print("\n[yellow]daemon stopped[/yellow]")


# ---------------------------------------------------------------------------
# skills hub (stage 13b)
# ---------------------------------------------------------------------------


@skills_app.command("search")
def skills_search(
    query: str = typer.Argument(..., help="Substring matched against name + description."),
    include_staged: bool = typer.Option(
        False, "--include-staged", "-S", help="Also search staged skills."
    ),
) -> None:
    """Search installed skills by name or description (case-insensitive)."""
    mgr = _skills_manager()
    skills = mgr.list(include_staged=include_staged)
    q = query.lower()
    hits = [
        s for s in skills
        if q in s.name.lower() or q in (s.description or "").lower()
    ]
    if not hits:
        console.print(f"[dim]No skills matched {query!r}.[/dim]")
        return
    table = Table(title=f"Skills matching {query!r}")
    table.add_column("Name", style="bold")
    table.add_column("Category")
    table.add_column("Description")
    table.add_column("Path")
    for s in hits:
        table.add_row(s.name, s.category, s.description, str(s.path))
    console.print(table)


@skills_app.command("install")
def skills_install(
    source: str = typer.Argument(
        ...,
        help=(
            "Source of the skill: an absolute / relative directory containing "
            "SKILL.md, a .zip / .tar.gz archive, or a git URL "
            "(detected by the .git suffix or a github.com / gitlab.com prefix)."
        ),
    ),
    name: str | None = typer.Option(
        None, "--name", help="Override the destination skill name (default: source basename)."
    ),
    category: str = typer.Option(
        "user", "--category", "-c", help="Category subdirectory under skills/."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing skill of the same name."
    ),
) -> None:
    """Install a skill from a local path, archive, or git URL."""
    from cytopert.skills.hub import install_from_source

    mgr = _skills_manager()
    try:
        path = install_from_source(
            mgr,
            source=source,
            name=name,
            category=category,
            force=force,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]\u2713[/green] Installed skill at {path}")


@skills_app.command("uninstall")
def skills_uninstall(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Remove an installed skill (mirror of `skills delete`, kept for hub UX symmetry)."""
    mgr = _skills_manager()
    try:
        mgr.view(name)  # existence probe
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    if not yes and not typer.confirm(f"Permanently delete skill {name!r}?"):
        raise typer.Exit()
    mgr.delete(name)
    console.print(f"[green]\u2713[/green] Uninstalled {name}.")


if __name__ == "__main__":
    app()

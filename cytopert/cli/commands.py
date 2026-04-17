"""CLI commands for CytoPert."""

import asyncio
import json
import os

import typer
from rich.console import Console
from rich.table import Table

from cytopert import __logo__, __version__

app = typer.Typer(
    name="cytopert",
    help=f"{__logo__} CytoPert - Cell perturbation differential response mechanism parsing",
    no_args_is_help=True,
)

memory_app = typer.Typer(name="memory", help="Manage CytoPert persistent memory.", no_args_is_help=True)
skills_app = typer.Typer(name="skills", help="Manage CytoPert procedural skills.", no_args_is_help=True)
chains_app = typer.Typer(name="chains", help="Inspect mechanism chain lifecycle.", no_args_is_help=True)
evidence_app = typer.Typer(name="evidence", help="Search persistent evidence (cross-session).", no_args_is_help=True)
app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
app.add_typer(chains_app, name="chains")
app.add_typer(evidence_app, name="evidence")

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"{__logo__} cytopert v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(None, "--version", "-v", callback=_version_callback, is_eager=True),
) -> None:
    """CytoPert - Interactive framework for cell perturbation mechanism parsing."""
    pass


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
    console.print("  3. Run a workflow: [cyan]cytopert run-workflow nfatc1_mammary[/cyan]")
    console.print("  4. Inspect learnings: [cyan]cytopert memory show / cytopert skills list[/cyan]")


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:default", "--session", "-s", help="Session ID"),
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
    )
    if message:
        async def run_once() -> None:
            response = await agent_loop.process_direct(message, session_id)
            console.print(f"\n{__logo__} {response}")

        asyncio.run(run_once())
    else:
        console.print(f"{__logo__} Interactive mode (Ctrl+C to exit)\n")
        console.print("[dim]Commands: /exit, /quit, /reset (clear history)[/dim]\n")

        async def run_interactive() -> None:
            while True:
                try:
                    user_input = console.input("[bold blue]You:[/bold blue] ")
                    if not user_input.strip():
                        continue
                    command = user_input.strip().lower()
                    if command in {"/exit", "/quit"}:
                        console.print("Goodbye!")
                        break
                    if command in {"/reset", "/new"}:
                        agent_loop.sessions.reset(session_id)
                        console.print("[green]✓[/green] Session cleared.")
                        continue
                    response = await agent_loop.process_direct(user_input, session_id)
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
    """Run a workflow scenario (plan -> confirm -> compute -> evidence -> mechanism chains)."""
    from cytopert.config.loader import load_config
    from cytopert.workflow.pipeline import get_scenario_config

    config = load_config()
    if not config.get_api_key():
        console.print("[red]Error: No API key configured.[/red]")
        raise typer.Exit(1)
    if scenario == "nfatc1_mammary":
        from cytopert.workflow.scenarios.nfatc1_mammary import run as run_nfatc1
        research_question = question or "Identify differential response mechanism for Nfatc1 in mammary development."
        result = run_nfatc1(research_question, scenario_config=get_scenario_config(config, scenario), feedback=feedback)
        console.print(f"\n{__logo__} [bold]Workflow result (nfatc1_mammary):[/bold]\n")
        console.print(result["response"])
        if feedback:
            console.print(f"\n[dim]Feedback applied: {feedback}[/dim]")
    else:
        console.print(f"[yellow]Unknown scenario: {scenario}. Known: nfatc1_mammary.[/yellow]")
        console.print("[dim]Add scenario config in workflow.scenarios or use nfatc1_mammary.[/dim]")


@app.command()
def status() -> None:
    """Show CytoPert status."""
    from cytopert.config.loader import get_config_path, load_config
    from cytopert.memory.store import MemoryStore
    from cytopert.persistence.chain_db import ChainStore
    from cytopert.persistence.evidence_db import EvidenceDB
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import (
        get_chains_dir,
        get_memory_dir,
        get_skills_dir,
        get_state_db_path,
    )

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path
    console.print(f"{__logo__} CytoPert Status\n")
    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")
    if config_path.exists():
        console.print(f"Model: {config.agents.defaults.model}")
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


if __name__ == "__main__":
    app()

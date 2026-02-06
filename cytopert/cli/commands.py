"""CLI commands for CytoPert."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from cytopert import __logo__, __version__

app = typer.Typer(
    name="cytopert",
    help=f"{__logo__} CytoPert - Cell perturbation differential response mechanism parsing",
    no_args_is_help=True,
)

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
    """Initialize CytoPert configuration and workspace."""
    from cytopert.config.loader import get_config_path, save_config
    from cytopert.config.schema import Config
    from cytopert.utils.helpers import get_workspace_path

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
    console.print(f"\n{__logo__} CytoPert is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.cytopert/config.json[/cyan]")
    console.print("     (e.g. providers.openrouter.apiKey from https://openrouter.ai/keys)")
    console.print("  2. Chat: [cyan]cytopert agent -m \"Your question\"[/cyan]")
    console.print("  3. Run a workflow: [cyan]cytopert run-workflow nfatc1_mammary[/cyan]")


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:default", "--session", "-s", help="Session ID"),
) -> None:
    """Interact with the agent (interactive or single message)."""
    from cytopert.config.loader import load_config
    from cytopert.providers.litellm_provider import LiteLLMProvider
    from cytopert.agent.loop import AgentLoop

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


if __name__ == "__main__":
    app()

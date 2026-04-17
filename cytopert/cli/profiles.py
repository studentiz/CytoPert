"""Profile management for CytoPert (stage 12).

CytoPert lets a researcher run several isolated configurations side by
side -- for example one profile per study, or one profile per remote
inference endpoint. Each profile gets its own complete tree under
``~/.cytopert/profiles/<name>/``: config.json, workspace, memory,
chains, skills, state.db, sessions, trajectories, plugins.

Three ways to switch profile, in priority order:

1. ``cytopert -p <name> ...``   -- one-shot env-var override for this
   process. Equivalent to ``CYTOPERT_HOME=~/.cytopert/profiles/<name>``.
2. ``cytopert profile use <name>`` -- writes
   ``~/.cytopert/active_profile`` so future invocations without ``-p``
   default to that profile.
3. (none)                       -- the default root ``~/.cytopert/``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cytopert.utils import helpers as hh

# Re-export wrappers so callers (and tests) get the runtime-resolved
# values; the previous "from ... import CYTOPERT_ROOT_DIR" pattern
# bound the constant once at module-load time and ignored later
# monkeypatching from the test fixture.
active_profile_name = hh.active_profile_name
set_active_profile = hh.set_active_profile

profile_app = typer.Typer(
    name="profile",
    help="List / create / delete / switch CytoPert profiles.",
    no_args_is_help=True,
)


@dataclass
class ProfileSummary:
    name: str
    path: Path
    has_config: bool
    is_active: bool


def list_profiles() -> list[ProfileSummary]:
    """Return one summary per profile directory under profiles_dir()."""
    base = hh.profiles_dir()
    active = hh.active_profile_name()
    out: list[ProfileSummary] = []
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        out.append(
            ProfileSummary(
                name=child.name,
                path=child,
                has_config=(child / "config.json").exists(),
                is_active=(child.name == active),
            )
        )
    return out


def _profile_path(name: str) -> Path:
    # Resolve via the helpers module attribute lookup so monkeypatching
    # ``hh.CYTOPERT_ROOT_DIR`` in tests propagates to this CLI layer.
    return hh.CYTOPERT_ROOT_DIR / hh.PROFILES_SUBDIR / name


@profile_app.command("list")
def profile_list() -> None:
    """List all profiles under ~/.cytopert/profiles."""
    console = Console()
    rows = list_profiles()
    if not rows:
        console.print("[dim]No profiles. Create one with `cytopert profile new <name>`.[/dim]")
        return
    table = Table(title="CytoPert profiles", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Active")
    table.add_column("Has config")
    table.add_column("Path")
    for r in rows:
        table.add_row(
            r.name,
            "[green]\u2713[/green]" if r.is_active else "",
            "[green]\u2713[/green]" if r.has_config else "[red]no[/red]",
            str(r.path),
        )
    console.print(table)


@profile_app.command("show")
def profile_show() -> None:
    """Print the currently active profile name (or none)."""
    console = Console()
    name = active_profile_name()
    if name is None:
        console.print("[dim]No active profile (using the default root).[/dim]")
        return
    console.print(f"Active profile: [bold]{name}[/bold]")


@profile_app.command("new")
def profile_new(name: str = typer.Argument(...)) -> None:
    """Create an empty profile directory (does NOT switch to it)."""
    console = Console()
    path = _profile_path(name)
    if path.exists():
        console.print(f"[yellow]Profile {name!r} already exists at {path}[/yellow]")
        raise typer.Exit(1)
    path.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[green]\u2713[/green] Created profile {name!r} at {path}.\n"
        f"Run [cyan]cytopert -p {name} setup[/cyan] to initialise it,\n"
        f"or [cyan]cytopert profile use {name}[/cyan] to make it the default."
    )


@profile_app.command("delete")
def profile_delete(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a profile and all of its data (irreversible)."""
    console = Console()
    path = _profile_path(name)
    if not path.exists():
        console.print(f"[red]No profile {name!r} at {path}[/red]")
        raise typer.Exit(1)
    if not yes and not typer.confirm(
        f"Permanently delete profile {name!r} ({path})?"
    ):
        raise typer.Exit()
    shutil.rmtree(path)
    if active_profile_name() == name:
        set_active_profile(None)
        console.print(
            "[yellow]Active profile removed; falling back to the default root.[/yellow]"
        )
    console.print(f"[green]\u2713[/green] Deleted profile {name!r}.")


@profile_app.command("use")
def profile_use(
    name: str = typer.Argument(
        None,
        help="Profile name to activate. Pass --clear (or omit name) to clear.",
    ),
    clear: bool = typer.Option(False, "--clear", help="Clear the active profile."),
) -> None:
    """Persist the default profile in ~/.cytopert/active_profile."""
    console = Console()
    if clear or name is None:
        set_active_profile(None)
        console.print("[green]\u2713[/green] Active profile cleared.")
        return
    if not _profile_path(name).exists():
        console.print(
            f"[red]No profile {name!r}; create it first with `cytopert profile new {name}`.[/red]"
        )
        raise typer.Exit(1)
    set_active_profile(name)
    console.print(
        f"[green]\u2713[/green] Active profile set to [bold]{name}[/bold]."
    )


__all__ = [
    "ProfileSummary",
    "list_profiles",
    "profile_app",
]

"""Slash-command dispatcher for the interactive ``cytopert agent`` shell.

The dispatcher is intentionally separate from the rest of the CLI so a
future prompt_toolkit-based shell (stage 11) can re-use the same command
table without dragging in the rich-console interactive loop.

Each handler returns one of three verdicts:
    * ``"handled"``    -- the slash command was recognised and the loop
                          should ask for the next user input.
    * ``"exit"``       -- the user wants to leave the shell.
    * ``"passthrough"``-- the input does not match any known slash
                          command; the loop should send it to the LLM
                          unchanged (so a literal ``/path/to/file`` does
                          not get swallowed silently).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:  # avoid circular import at module load time
    from cytopert.agent.loop import AgentLoop


SLASH_COMMAND_HELP: list[tuple[str, str]] = [
    ("/help", "Show this help."),
    ("/exit, /quit", "Leave the interactive shell."),
    ("/reset, /new", "Clear conversation + re-arm plan gate + reset compressor."),
    ("/skip-plan", "Disable the plan gate for this session."),
    ("/model [name]", "Show or switch model in-process for this session."),
    ("/usage", "Show this session's accumulated token / cost stats."),
    ("/history [N]", "Print the last N user / assistant messages (default 6)."),
    ("/skills", "List installed skills."),
    ("/chains", "List recent mechanism chains."),
    ("/retry", "Re-send the last user message."),
    ("/undo", "Drop the last user + assistant turn from this session."),
]


def _print_help(console: Console) -> None:
    table = Table(title="Slash commands", show_lines=False)
    table.add_column("Command", style="bold")
    table.add_column("What it does")
    for cmd, desc in SLASH_COMMAND_HELP:
        table.add_row(cmd, desc)
    console.print(table)


def _show_usage(loop: "AgentLoop", session_id: str, console: Console) -> None:
    sess = loop.sessions.get_or_create(session_id)
    stats = sess.metadata.get("usage") or {}
    if not stats:
        console.print("[dim]No usage recorded for this session yet.[/dim]")
        return
    console.print(
        f"calls={stats.get('calls', 0)}  "
        f"prompt={stats.get('prompt_tokens', 0)}  "
        f"completion={stats.get('completion_tokens', 0)}  "
        f"cost_usd={stats.get('cost_usd', 0.0):.6f}"
    )


def _show_history(loop: "AgentLoop", session_id: str, n: int, console: Console) -> None:
    sess = loop.sessions.get_or_create(session_id)
    if not sess.messages:
        console.print("[dim]Empty session.[/dim]")
        return
    for msg in sess.messages[-max(1, n):]:
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))
        if len(content) > 400:
            content = content[:400] + "..."
        console.print(f"[bold]{role}[/bold]: {content}")


def _show_skills(console: Console) -> None:
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import get_skills_dir

    skills = SkillsManager(get_skills_dir()).list()
    if not skills:
        console.print("[dim]No skills installed.[/dim]")
        return
    for s in skills:
        console.print(f"- [{s.category}] {s.name} -- {s.description}")


def _show_chains(console: Console) -> None:
    from cytopert.persistence.chain_db import ChainStore
    from cytopert.utils.helpers import get_chains_dir, get_state_db_path

    rows = ChainStore(get_state_db_path(), get_chains_dir()).list(limit=10)
    if not rows:
        console.print("[dim]No chains yet.[/dim]")
        return
    for chain, status in rows:
        console.print(f"{chain.id}  [{status}]  P={chain.priority}  -- {chain.summary[:80]}")


def _switch_model(loop: "AgentLoop", new_model: str, console: Console) -> None:
    loop.model = new_model
    if loop.context_engine is not None:
        try:
            loop.context_engine.update_model(
                new_model,
                getattr(loop.context_engine, "context_length", 32768),
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]ContextEngine update_model failed: {exc}[/yellow]")
    console.print(f"[green]\u2713[/green] Model switched in-process to {new_model!r}.")
    console.print(
        "[dim]This switch only affects the current shell. To persist, "
        "run [bold]cytopert config set agents.defaults.model "
        f"{new_model}[/bold].[/dim]"
    )


def _retry(loop: "AgentLoop", session_id: str, console: Console) -> str:
    """Surface the last user message so the loop re-sends it.

    Returns the message body so the caller can dispatch it to the LLM.
    The caller then either gets back ``"handled"`` (no last message) or
    ``"passthrough_with_text"`` -- represented here by writing the text
    into ``loop._retry_message`` (a tiny attribute-stash) so the
    handler can be invoked from the slash dispatcher without growing a
    new return type. We avoid that complexity by returning ``"handled"``
    when there is no message and prompting the user otherwise.
    """
    sess = loop.sessions.get_or_create(session_id)
    user_msgs = [m for m in sess.messages if m.get("role") == "user"]
    if not user_msgs:
        console.print("[dim]No previous user message to retry.[/dim]")
        return "handled"
    last = user_msgs[-1]
    console.print(
        f"[dim]Retrying last user message ({len(last.get('content','')[:60])} chars):[/dim]"
    )
    console.print(f"  {last.get('content', '')[:200]}")
    # Drop the previous assistant reply so the retry stands on its own
    # turn-pair, then re-emit the user message via _retry_message.
    if sess.messages and sess.messages[-1].get("role") == "assistant":
        sess.messages.pop()
        loop.sessions.save(sess)
    loop._retry_message = last.get("content", "")  # consumed by the loop
    return "handled"


def _undo(loop: "AgentLoop", session_id: str, console: Console) -> None:
    sess = loop.sessions.get_or_create(session_id)
    dropped = 0
    while sess.messages and dropped < 2:
        sess.messages.pop()
        dropped += 1
    loop.sessions.save(sess)
    console.print(
        f"[green]\u2713[/green] Dropped the last {dropped} message(s) from this session."
    )


def _toggle_skip_plan(loop: "AgentLoop", session_id: str, console: Console) -> None:
    from cytopert.agent.loop import PLAN_MODE_DISABLED, PLAN_MODE_KEY

    sess = loop.sessions.get_or_create(session_id)
    sess.metadata[PLAN_MODE_KEY] = PLAN_MODE_DISABLED
    loop.sessions.save(sess)
    console.print("[green]\u2713[/green] Plan gate disabled for this session.")


def _reset(loop: "AgentLoop", session_id: str, console: Console) -> None:
    loop.sessions.reset(session_id)
    loop.enable_plan_gate(session_id)
    if loop.context_engine is not None:
        try:
            loop.context_engine.on_session_reset()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]ContextEngine reset failed: {exc}[/yellow]")
    console.print("[green]\u2713[/green] Session cleared; plan gate re-armed.")


def handle_slash_command(
    raw: str,
    loop: "AgentLoop",
    session_id: str,
    console: Console,
) -> str:
    """Dispatch a single slash command. See module docstring for verdicts."""
    parts = raw.strip().split()
    if not parts:
        return "passthrough"
    head = parts[0].lower()
    args = parts[1:]

    if head in {"/exit", "/quit"}:
        console.print("Goodbye!")
        return "exit"
    if head in {"/reset", "/new"}:
        _reset(loop, session_id, console)
        return "handled"
    if head == "/skip-plan":
        _toggle_skip_plan(loop, session_id, console)
        return "handled"
    if head == "/help":
        _print_help(console)
        return "handled"
    if head == "/usage":
        _show_usage(loop, session_id, console)
        return "handled"
    if head == "/history":
        try:
            n = int(args[0]) if args else 6
        except ValueError:
            n = 6
        _show_history(loop, session_id, n, console)
        return "handled"
    if head == "/skills":
        _show_skills(console)
        return "handled"
    if head == "/chains":
        _show_chains(console)
        return "handled"
    if head == "/model":
        if not args:
            console.print(f"Current model: [bold]{loop.model}[/bold]")
            return "handled"
        _switch_model(loop, args[0], console)
        return "handled"
    if head == "/retry":
        return _retry(loop, session_id, console)
    if head == "/undo":
        _undo(loop, session_id, console)
        return "handled"

    # Unknown slash command: pass through unchanged so an LLM prompt
    # like "/path/to/data.h5ad explain" survives the gate.
    return "passthrough"


__all__ = ["SLASH_COMMAND_HELP", "handle_slash_command"]

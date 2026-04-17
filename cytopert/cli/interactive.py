"""Prompt-toolkit-based interactive shell for ``cytopert agent``.

Adapted in spirit from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:cli.py (~10000 lines, MIT
License). This module is ~250 lines because it only borrows the
PromptSession + key-bindings + bottom-toolbar pattern; we delegate
slash-command dispatch to ``cytopert.cli.interactive_slash`` so future
front-ends can re-use the table.

Features
--------
* Persistent history at ``~/.cytopert/history.txt``.
* Multiline editing via ``Esc + Enter``; plain ``Enter`` submits.
* WordCompleter on slash commands (typed ``/`` triggers a popup).
* Streaming assistant text via ``rich.live.Live`` panel.
* Tool round-trips render an inline ``calling tool: <name>`` line.
* Bottom toolbar shows model / provider / plan-mode / cumulative
  token usage / cost / evidence count / chains count.
* ``Ctrl+C`` once cancels the in-flight turn (sets an
  ``asyncio.Event`` the AgentLoop checks between iterations); twice
  exits the shell.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from cytopert import __logo__
from cytopert.cli.interactive_slash import (
    SLASH_COMMAND_HELP,
    handle_slash_command,
)

if TYPE_CHECKING:
    from cytopert.agent.loop import AgentLoop


def _slash_words() -> list[str]:
    """Flatten the SLASH_COMMAND_HELP table into a completer word list."""
    out: list[str] = []
    for cmd, _ in SLASH_COMMAND_HELP:
        for token in cmd.split(","):
            token = token.strip()
            if token.startswith("/"):
                head = token.split()[0]
                if head not in out:
                    out.append(head)
    return out


def _render_toolbar(agent_loop: "AgentLoop", session_id: str) -> str:
    """Return the bottom-toolbar text refreshed once per keypress."""
    sess = agent_loop.sessions.get_or_create(session_id)
    usage = sess.metadata.get("usage") or {}
    plan_mode = sess.metadata.get("plan_mode", "disabled")
    try:
        evi = agent_loop.evidence_db.count()
    except Exception:  # noqa: BLE001
        evi = 0
    try:
        chains_total = (
            agent_loop.chains.count("proposed")
            + agent_loop.chains.count("supported")
            + agent_loop.chains.count("refuted")
            + agent_loop.chains.count("superseded")
        )
    except Exception:  # noqa: BLE001
        chains_total = 0
    cost = usage.get("cost_usd", 0.0) or 0.0
    return (
        f"  model={agent_loop.model}  plan={plan_mode}  "
        f"calls={usage.get('calls', 0)}  "
        f"prompt={usage.get('prompt_tokens', 0)}  "
        f"completion={usage.get('completion_tokens', 0)}  "
        f"cost=${cost:.6f}  evi={evi}  chains={chains_total}  "
        "(Esc+Enter for newline; Ctrl+C cancels turn or exits)"
    )


async def run_prompt_toolkit_shell(
    agent_loop: "AgentLoop",
    session_id: str,
    initial_feedback: str | None = None,
) -> None:
    """Run the prompt_toolkit-based interactive shell to completion."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.patch_stdout import patch_stdout
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "prompt_toolkit is required for the interactive shell; "
            "install with `pip install prompt_toolkit>=3.0`."
        ) from exc

    from cytopert.utils.helpers import get_data_path

    console = Console()
    history_path = str(get_data_path() / "history.txt")
    interrupt_event = asyncio.Event()
    feedback = initial_feedback

    # Cancel-then-exit semantics: first Ctrl+C cancels the in-flight
    # turn; a second Ctrl+C while no turn is running exits.
    in_flight = {"value": False}

    bindings = KeyBindings()

    @bindings.add("c-c", eager=True)
    def _interrupt(event):  # noqa: ARG001 -- prompt_toolkit signature
        if in_flight["value"]:
            interrupt_event.set()
            console.print("\n[yellow][Cancelling current turn...][/yellow]")
        else:
            event.app.exit(exception=KeyboardInterrupt)

    completer = WordCompleter(_slash_words(), ignore_case=True, sentence=False)

    session: PromptSession = PromptSession(
        history=FileHistory(history_path),
        completer=completer,
        multiline=False,
        key_bindings=bindings,
        bottom_toolbar=lambda: _render_toolbar(agent_loop, session_id),
    )

    # Plan gate is OFF by default. The previous "ON by default" policy
    # forced every casual message ("hello", "what can you do?") into a
    # plan-only turn with no tools, which produced repetitive
    # "I need data first" replies that ignored the user's actual
    # message. Power users can opt back in at any time with
    # `/plan-gate on`.

    console.print(f"{__logo__} CytoPert interactive shell")
    console.print(
        "[dim]Type [bold]/help[/bold] for slash commands. "
        "Plan-gate is OFF by default (chat freely; tools run as needed). "
        "Use [bold]/plan-gate on[/bold] to require a textual plan + 'go' "
        "before tools fire. Press [bold]Ctrl+C[/bold] once to cancel the "
        "current turn, twice to exit.[/dim]\n"
    )

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async("You: ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            return

        if not user_input.strip():
            continue

        if user_input.lstrip().startswith("/"):
            verdict = handle_slash_command(user_input, agent_loop, session_id, console)
            if verdict == "exit":
                return
            if verdict == "handled":
                continue
            # passthrough: fall through to LLM dispatch

        interrupt_event.clear()
        in_flight["value"] = True
        try:
            await _stream_one_turn(
                console=console,
                agent_loop=agent_loop,
                session_id=session_id,
                user_input=user_input,
                feedback=feedback,
                interrupt_event=interrupt_event,
            )
            feedback = None
        finally:
            in_flight["value"] = False


async def _stream_one_turn(
    *,
    console: Console,
    agent_loop: "AgentLoop",
    session_id: str,
    user_input: str,
    feedback: str | None,
    interrupt_event: asyncio.Event,
) -> None:
    """Run a single turn against the agent and stream its assistant text."""
    buffer: list[str] = []
    tool_log: list[str] = []

    def _on_text(delta: str) -> None:
        buffer.append(delta)

    def _on_tool(kind: str, name: str, payload: str) -> None:
        if kind == "start":
            tool_log.append(f"[blue]calling tool: {name}({payload})[/blue]")
        else:
            tool_log.append(
                f"[green]  -> {name}: {payload[:80]}"
                + ("..." if len(payload) > 80 else "")
                + "[/green]"
            )

    def _render_panel() -> Panel:
        body_lines = list(tool_log)
        body_lines.append("")
        body_lines.append("".join(buffer))
        return Panel("\n".join(body_lines), title=f"{__logo__} CytoPert")

    with Live(_render_panel(), console=console, refresh_per_second=8) as live:

        def _on_text_with_render(delta: str) -> None:
            _on_text(delta)
            live.update(_render_panel())

        def _on_tool_with_render(kind: str, name: str, payload: str) -> None:
            _on_tool(kind, name, payload)
            live.update(_render_panel())

        final = await agent_loop.process_direct(
            user_input,
            session_id,
            user_feedback=feedback,
            stream_callback=_on_text_with_render,
            interrupt_event=interrupt_event,
            on_tool_event=_on_tool_with_render,
        )
        # If the streamed buffer is empty (e.g. the model emitted only
        # tool_calls, or the binding-enforcer rewrote the reply offline),
        # paint the canonical final_content into the panel so the user
        # always sees it before we exit Live.
        if not "".join(buffer).strip() and final:
            buffer.append(final)
            live.update(_render_panel())
    # If evidence-binding enforcement appended an advisory the streamed
    # buffer never saw, surface the diff after the Live panel closes so
    # nothing is lost.
    streamed = "".join(buffer)
    if final and final not in streamed:
        diff = final[len(streamed):] if final.startswith(streamed) else final
        console.print(diff)


__all__ = ["run_prompt_toolkit_shell"]

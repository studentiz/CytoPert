"""Stage 11 tests for AgentLoop interrupt-and-redirect via interrupt_event."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cytopert.agent.loop import AgentLoop
from cytopert.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _SlowProvider(LLMProvider):
    """A provider whose chat coroutine cooperates with cancellation.

    Always returns a tool_call so the AgentLoop loop iterates; the
    sleep before each return gives the test room to flip the
    interrupt_event between iterations.
    """

    def __init__(self) -> None:
        super().__init__(api_key="x")
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
        stream_callback=None,
    ) -> LLMResponse:
        self.calls += 1
        await asyncio.sleep(0)
        return LLMResponse(
            content=f"iter-{self.calls}",
            tool_calls=[ToolCallRequest(id=f"tc-{self.calls}", name="skills_list", arguments={})],
            finish_reason="tool_calls",
        )

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYTOPERT_HOME", str(tmp_path / "cytopert_home"))
    yield


def test_interrupt_event_short_circuits_loop(tmp_path) -> None:
    """If interrupt_event is set, the AgentLoop returns the cancellation message."""

    provider = _SlowProvider()
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path / "ws",
        model="stub",
        enable_reflection=False,
        load_plugins=False,
    )

    ev = asyncio.Event()
    ev.set()  # already cancelled before the first iteration

    out = asyncio.run(
        loop.process_direct(
            "anything",
            session_key="t",
            interrupt_event=ev,
        )
    )
    assert "[Cancelled]" in out, out
    # The loop must NOT have called the provider when the event was
    # already set before iteration 1.
    assert provider.calls == 0


def test_stream_callback_is_invoked_with_text_deltas(tmp_path) -> None:
    """A streaming-aware provider feeds chunks into the supplied callback."""

    pieces = ["Hello, ", "this is ", "streamed."]

    class _Stream(LLMProvider):
        def __init__(self) -> None:
            super().__init__(api_key="x")

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            api_base: str | None = None,
            stream_callback=None,
        ) -> LLMResponse:
            full = ""
            for piece in pieces:
                full += piece
                if stream_callback is not None:
                    stream_callback(piece)
            return LLMResponse(content=full, finish_reason="stop")

        def get_default_model(self) -> str:
            return "stub"

    loop = AgentLoop(
        provider=_Stream(),
        workspace=tmp_path / "ws",
        model="stub",
        enable_reflection=False,
        load_plugins=False,
    )

    received: list[str] = []
    out = asyncio.run(
        loop.process_direct(
            "stream please",
            session_key="t",
            stream_callback=received.append,
        )
    )
    assert "".join(received) == "Hello, this is streamed."
    assert out == "Hello, this is streamed."


def test_on_tool_event_fires_for_each_dispatch(tmp_path) -> None:
    """on_tool_event(start, name, args) and (result, name, payload) are called."""

    class _OneToolThenDone(LLMProvider):
        def __init__(self) -> None:
            super().__init__(api_key="x")
            self.calls = 0

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            api_base: str | None = None,
            stream_callback=None,
        ) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(id="t1", name="skills_list", arguments={})
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResponse(content="done", finish_reason="stop")

        def get_default_model(self) -> str:
            return "stub"

    loop = AgentLoop(
        provider=_OneToolThenDone(),
        workspace=tmp_path / "ws",
        model="stub",
        enable_reflection=False,
        load_plugins=False,
    )

    events: list[tuple[str, str, str]] = []

    asyncio.run(
        loop.process_direct(
            "do a tool",
            session_key="t",
            on_tool_event=lambda kind, name, payload: events.append((kind, name, payload)),
        )
    )
    kinds = [e[0] for e in events]
    names = [e[1] for e in events]
    assert "start" in kinds and "result" in kinds
    assert "skills_list" in names

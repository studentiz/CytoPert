"""Stage 1.B regression tests for the upgraded ToolRegistry."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from cytopert.agent.tools.base import Tool
from cytopert.agent.tools.registry import (
    ToolRegistry,
    tool_error,
    tool_result,
)


class _PingTool(Tool):
    @property
    def name(self) -> str:
        return "ping"

    @property
    def description(self) -> str:
        return "Ping echo"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}},
                "required": []}

    async def execute(self, text: str = "pong") -> str:
        return f"pong:{text}"


def test_register_function_and_dispatch() -> None:
    reg = ToolRegistry()

    async def echo(text: str = "hi") -> str:
        return tool_result(echo=text)

    reg.register_function(
        name="echo",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=echo,
        description="echo back",
    )
    assert "echo" in reg
    out = asyncio.run(reg.execute("echo", {"text": "world"}))
    assert json.loads(out) == {"echo": "world"}


def test_register_function_rejects_sync_handler() -> None:
    reg = ToolRegistry()

    def sync_handler(**_: object) -> str:
        return "nope"

    with pytest.raises(TypeError):
        reg.register_function(
            name="sync", schema={"type": "object"}, handler=sync_handler  # type: ignore[arg-type]
        )


def test_shadowing_is_rejected() -> None:
    reg = ToolRegistry()
    reg.register(_PingTool())
    reg.register(_PingTool())  # second register on same name is a no-op
    assert len([n for n in reg.tool_names if n == "ping"]) == 1


def test_check_fn_filters_definitions_but_not_dispatch() -> None:
    reg = ToolRegistry()

    async def echo(**_: object) -> str:
        return "ok"

    reg.register_function(
        name="echo", schema={"type": "object"}, handler=echo,
        check_fn=lambda: False,
    )
    # Schema list filters the entry out
    assert reg.get_definitions() == []
    # ...but dispatch still works (callers that already have the name).
    assert asyncio.run(reg.execute("echo", {})) == "ok"


def test_unknown_tool_returns_error_string() -> None:
    reg = ToolRegistry()
    assert "Error" in asyncio.run(reg.execute("missing", {}))


def test_deregister_removes_tool_and_definitions() -> None:
    reg = ToolRegistry()
    reg.register(_PingTool())
    assert "ping" in reg
    reg.deregister("ping")
    assert "ping" not in reg
    assert reg.get_definitions() == []


def test_concurrent_register_and_dispatch_no_deadlock() -> None:
    reg = ToolRegistry()
    reg.register(_PingTool())
    errors: list[str] = []

    def _spam_register() -> None:
        for i in range(20):
            try:
                reg.register_function(
                    name=f"plug_{i}",
                    schema={"type": "object"},
                    handler=_async_ok,
                )
            except Exception as exc:
                errors.append(repr(exc))

    async def _async_ok(**_: object) -> str:
        return "ok"

    def _spam_dispatch() -> None:
        loop = asyncio.new_event_loop()
        try:
            for _ in range(20):
                loop.run_until_complete(reg.execute("ping", {"text": "x"}))
        finally:
            loop.close()

    threads = [threading.Thread(target=_spam_register), threading.Thread(target=_spam_dispatch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert any(name.startswith("plug_") for name in reg.tool_names)


def test_helpers_serialize_to_json() -> None:
    assert json.loads(tool_error("bad")) == {"error": "bad"}
    assert json.loads(tool_error("bad", code=404)) == {"error": "bad", "code": 404}
    assert json.loads(tool_result(success=True, n=5)) == {"success": True, "n": 5}
    assert json.loads(tool_result({"a": 1})) == {"a": 1}

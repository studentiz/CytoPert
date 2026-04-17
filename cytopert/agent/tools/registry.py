"""Thread-safe tool registry for CytoPert agent tools.

Adapted from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:tools/registry.py (MIT License).
See docs/hermes-borrowing.md for the per-module diff rationale.

Differences from upstream:
    * CytoPert tools are stateful (workspace, EvidenceDB, MemoryStore, ...)
      and therefore are wired in by ``AgentLoop._register_default_tools``
      instead of via top-level ``registry.register(...)`` self-registration.
      The ``register_function`` API and ``discover_self_registering_tools``
      helper are still provided so future plugin packages (stage 7.3) can
      use the hermes-style import-time registration pattern.
    * No MCP toolset alias machinery (CytoPert does not speak MCP).
    * No per-tool ``max_result_size_chars`` budget (CytoPert relies on the
      ContextCompressor in stage 4.C instead).
    * ``execute`` stays an ``async`` coroutine because the entire CytoPert
      ``AgentLoop`` runs on asyncio.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from cytopert.agent.tools.base import Tool

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    """Metadata for a single registered tool.

    Holds either a ``Tool`` subclass instance (the legacy CytoPert path) or
    a function-based handler with an explicit schema. ``handler`` for the
    function path is a coroutine ``handler(**params) -> str``.
    """

    name: str
    toolset: str = "default"
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    tool: Tool | None = None
    handler: Callable[..., Awaitable[str]] | None = None
    check_fn: Callable[[], bool] | None = None
    requires_env: list[str] = field(default_factory=list)


class ToolRegistry:
    """Registry for CytoPert agent tools, safe for concurrent reads.

    The registry is consumed by ``AgentLoop`` for tool dispatch and by
    plugin discovery (stage 7.3) for late registration. Schema enumeration
    and dispatch take an internal RLock so a plugin reload thread cannot
    mutate the registry while ``AgentLoop`` is iterating it.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        # RLock (not Lock) because dispatch may re-enter via tool handlers
        # that themselves call registry methods (e.g. introspection from a
        # meta-tool). A plain Lock would deadlock in that path.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool, *, toolset: str = "default") -> None:
        """Register a CytoPert ``Tool`` instance (legacy path used by ``AgentLoop``).

        Existing-name collisions are rejected with an error log; callers must
        ``deregister`` first if a swap is intentional. This mirrors hermes'
        anti-shadowing rule and prevents a noisy plugin from silently
        overriding a built-in tool.
        """
        with self._lock:
            existing = self._tools.get(tool.name)
            if existing is not None:
                logger.error(
                    "Tool registration REJECTED: %r (toolset=%r) would shadow "
                    "existing tool from toolset %r. Call deregister first.",
                    tool.name,
                    toolset,
                    existing.toolset,
                )
                return
            self._tools[tool.name] = ToolEntry(
                name=tool.name,
                toolset=toolset,
                description=tool.description,
                schema=tool.parameters,
                tool=tool,
            )

    def register_function(
        self,
        name: str,
        schema: dict[str, Any],
        handler: Callable[..., Awaitable[str]],
        *,
        toolset: str = "default",
        description: str = "",
        check_fn: Callable[[], bool] | None = None,
        requires_env: list[str] | None = None,
    ) -> None:
        """Register a function-based tool (used by plugins in stage 7.3).

        The handler must be an async coroutine returning a JSON string.
        ``schema`` follows JSON Schema (the OpenAI ``function.parameters``
        shape), and ``check_fn`` -- when supplied -- is a cheap callable
        that decides at every prompt-build whether the tool should be
        exposed to the model.
        """
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(
                f"Tool handler {name!r} must be an async coroutine; "
                f"got {type(handler).__name__}"
            )
        with self._lock:
            if name in self._tools:
                logger.error(
                    "Tool registration REJECTED: %r already registered "
                    "(toolset=%r). Call deregister first.",
                    name,
                    self._tools[name].toolset,
                )
                return
            self._tools[name] = ToolEntry(
                name=name,
                toolset=toolset,
                description=description or schema.get("description", ""),
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=list(requires_env or []),
            )

    def deregister(self, name: str) -> None:
        """Remove a tool. No-op if missing. Used by plugin reloads."""
        with self._lock:
            self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the underlying ``Tool`` instance, or ``None`` if missing.

        For function-based tools this returns ``None`` even though the tool
        is dispatchable -- legacy callers that want the ``Tool`` object are
        only relevant for built-in tools registered via ``register``.
        """
        with self._lock:
            entry = self._tools.get(name)
            return entry.tool if entry is not None else None

    def get_entry(self, name: str) -> ToolEntry | None:
        """Return the full registry entry (tool, handler, schema, metadata)."""
        with self._lock:
            return self._tools.get(name)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    @property
    def tool_names(self) -> list[str]:
        with self._lock:
            return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-format function schemas for all registered tools.

        Tools whose ``check_fn`` returns False (or raises) are filtered out;
        this lets plugins gate availability on runtime conditions like API
        keys without unregistering the tool entirely.
        """
        defs: list[dict[str, Any]] = []
        with self._lock:
            entries = list(self._tools.values())
        for entry in entries:
            if entry.check_fn is not None:
                try:
                    if not entry.check_fn():
                        continue
                except Exception as exc:  # check_fn must never crash dispatch
                    logger.debug(
                        "check_fn for tool %r raised %s; hiding tool",
                        entry.name,
                        exc,
                    )
                    continue
            if entry.tool is not None:
                defs.append(entry.tool.to_schema())
            else:
                # Function-based tool: synthesize the OpenAI envelope.
                defs.append({
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.description,
                        "parameters": entry.schema,
                    },
                })
        return defs

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a registered tool. Always returns a string (JSON or plain).

        Errors are wrapped as ``Error: ...`` strings rather than raised, so
        the agent loop can append them as tool results without crashing the
        whole turn. The legacy CytoPert format ``Error executing X: msg`` is
        preserved for backward compatibility with existing tests and
        ``_enforce_evidence_gate`` heuristics.
        """
        entry = self.get_entry(name)
        if entry is None:
            return f"Error: Tool '{name}' not found"
        try:
            if entry.tool is not None:
                errors = entry.tool.validate_params(params)
                if errors:
                    return (
                        f"Error: Invalid parameters for tool '{name}': "
                        + "; ".join(errors)
                    )
                return await entry.tool.execute(**params)
            # Function-based tool: skip schema validation (plugins are
            # expected to validate inside their handler).
            assert entry.handler is not None
            return await entry.handler(**params)
        except Exception as exc:
            logger.exception("Tool %s dispatch error", name)
            return f"Error executing {name}: {exc}"

    # ------------------------------------------------------------------
    # Self-register discovery (used by plugin packages in stage 7.3)
    # ------------------------------------------------------------------

    def discover_self_registering_tools(
        self,
        package: str,
        tools_dir: Path,
        skip: tuple[str, ...] = ("__init__", "registry", "base"),
    ) -> list[str]:
        """Import every module in ``tools_dir`` whose top-level body calls
        ``registry.register*``, and return the list of imported modules.

        The AST gate prevents importing helper modules that happen to call
        ``registry.register`` from inside a function (and thus would not
        actually register anything at import time).
        """
        if not tools_dir.is_dir():
            return []
        imported: list[str] = []
        for path in sorted(tools_dir.glob("*.py")):
            if path.stem in skip:
                continue
            if not _module_top_level_registers(path):
                continue
            mod_name = f"{package}.{path.stem}"
            try:
                importlib.import_module(mod_name)
                imported.append(mod_name)
            except Exception as exc:
                logger.warning("Could not import tool module %s: %s", mod_name, exc)
        return imported


def _module_top_level_registers(path: Path) -> bool:
    """Return True iff the module's body contains ``registry.register*(...)``.

    Static AST inspection avoids importing helper modules that only call
    ``registry`` from inside a function. Mirrors the hermes pattern but
    with no symlink resolution because CytoPert tools live inside the
    package.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in {"register", "register_function"}
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "registry"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Tool result helpers
# ---------------------------------------------------------------------------
# All CytoPert tool handlers return strings. These helpers eliminate the
# ``json.dumps({"error": ...}, ensure_ascii=False)`` boilerplate that
# otherwise accumulates inside every tool implementation. They also make
# the calling convention obvious for plugin authors.


def tool_error(message: str, **extra: Any) -> str:
    """Format an error payload as a JSON string for a tool handler.

    Keeps the legacy ``Error: ...`` plain-text format reachable for callers
    that still want it, but most plugins should prefer the structured form
    so the agent loop can reason about ``error`` programmatically.
    """
    payload: dict[str, Any] = {"error": str(message)}
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def tool_result(data: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Format a success payload as a JSON string. Pass ``data`` *or* kwargs."""
    if data is not None and kwargs:
        raise ValueError("Pass either data dict or keyword arguments, not both")
    return json.dumps(data if data is not None else kwargs, ensure_ascii=False)


# Module-level singleton -- mirrors hermes' ``registry`` symbol so plugin
# packages (stage 7.3) can ``from cytopert.agent.tools.registry import registry``
# and call ``registry.register_function(...)`` at import time.
registry = ToolRegistry()

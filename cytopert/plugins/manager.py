"""Plugin discovery for CytoPert.

Adapted (significantly trimmed) from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:hermes_cli/plugins.py (MIT).
See docs/hermes-borrowing.md for the per-module diff rationale.

Three sources are searched, in order:
    1. ``~/.cytopert/plugins/<name>/cytopert_plugin.py`` -- user-level.
    2. ``./.cytopert/plugins/<name>/cytopert_plugin.py`` -- per-project,
       discovered relative to the active workspace.
    3. ``importlib.metadata.entry_points(group="cytopert.tools")`` and
       ``... group="cytopert.scenarios"`` -- pip-installed packages.

Each plugin module must expose a ``setup(ctx: PluginContext) -> None``
callable. ``ctx.register_tool`` registers a tool against the live
``ToolRegistry``; ``ctx.register_scenario`` registers a scenario factory
against ``cytopert.workflow.pipeline.SCENARIO_REGISTRY``.

Differences from upstream:
    * No lifecycle hooks (no on_session_start / on_message_pre etc.) --
      CytoPert plugins extend tools / scenarios only.
    * No callbacks (no clarify / sudo / approval) -- single-cell tools
      do not need execution-time approval prompts.
    * No CLI command registration via plugins -- the plugin surface is
      tools + scenarios only; if a plugin needs CLI hooks it should
      ship its own console_script.
    * ``disabled.txt`` (one plugin name per line) lets users opt-out
      of a plugin without uninstalling it.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cytopert.agent.tools.registry import ToolRegistry
from cytopert.utils.helpers import ensure_dir, get_data_path

if TYPE_CHECKING:
    from cytopert.workflow.pipeline import Pipeline

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_NAME = "cytopert_plugin.py"
PLUGIN_ENTRYPOINT_GROUPS = ("cytopert.tools", "cytopert.scenarios")
DEFAULT_DISABLED_FILE = "disabled.txt"


class PluginSource(str, Enum):
    """Where a discovered plugin came from."""

    USER = "user"
    PROJECT = "project"
    ENTRY_POINT = "entry_point"


@dataclass
class PluginInfo:
    """Bookkeeping for a single discovered plugin."""

    name: str
    source: PluginSource
    location: str
    setup: Callable[["PluginContext"], None] | None = None
    error: str | None = None
    enabled: bool = True
    tools_registered: list[str] = field(default_factory=list)
    scenarios_registered: list[str] = field(default_factory=list)


@dataclass
class PluginContext:
    """Context object handed to each plugin's ``setup`` callable."""

    info: PluginInfo
    registry: ToolRegistry
    workspace: Path
    evidence_db: Any
    memory: Any
    chain_store: Any

    def register_tool(
        self,
        name: str,
        schema: dict[str, Any],
        handler: Callable[..., Awaitable[str]],
        *,
        toolset: str | None = None,
        description: str = "",
        check_fn: Callable[[], bool] | None = None,
        requires_env: list[str] | None = None,
    ) -> None:
        """Register a function-style tool. Mirrors ``ToolRegistry.register_function``."""
        toolset_label = toolset or f"plugin:{self.info.name}"
        self.registry.register_function(
            name=name,
            schema=schema,
            handler=handler,
            toolset=toolset_label,
            description=description,
            check_fn=check_fn,
            requires_env=requires_env,
        )
        self.info.tools_registered.append(name)

    def register_scenario(
        self,
        name: str,
        factory: Callable[[], "Pipeline"],
    ) -> None:
        """Register a workflow scenario via SCENARIO_REGISTRY."""
        from cytopert.workflow.pipeline import register_scenario

        register_scenario(name, factory)
        self.info.scenarios_registered.append(name)


class PluginManager:
    """Discover and run plugin ``setup`` callables.

    Constructed once per AgentLoop. ``discover()`` returns the list of
    plugins that were found (regardless of whether their setup ran);
    ``setup_all(ctx_factory)`` runs the setup callable for every
    enabled plugin and returns the list of ``PluginInfo`` records with
    success / error annotations.
    """

    def __init__(
        self,
        *,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
        disabled_file: str = DEFAULT_DISABLED_FILE,
    ) -> None:
        self.user_dir = ensure_dir(
            user_dir if user_dir is not None else get_data_path() / "plugins"
        )
        self.project_dir = (
            project_dir.resolve()
            if project_dir is not None
            else None
        )
        self.disabled_file = disabled_file

    # ----- Discovery -------------------------------------------------

    def _disabled_set(self) -> set[str]:
        path = self.user_dir / self.disabled_file
        if not path.exists():
            return set()
        try:
            return {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return set()

    def _discover_directory(
        self, base: Path, source: PluginSource
    ) -> list[PluginInfo]:
        if not base.exists():
            return []
        out: list[PluginInfo] = []
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith((".", "_")):
                continue
            entry = child / PLUGIN_ENTRY_NAME
            if not entry.is_file():
                continue
            out.append(
                PluginInfo(name=child.name, source=source, location=str(entry))
            )
        return out

    def _discover_entry_points(self) -> list[PluginInfo]:
        out: list[PluginInfo] = []
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover - python 3.11+ guaranteed elsewhere
            return out
        seen_names: set[str] = set()
        for group in PLUGIN_ENTRYPOINT_GROUPS:
            try:
                eps = entry_points(group=group)
            except TypeError:
                # Older importlib.metadata API on some 3.10 patches.
                eps = entry_points().get(group, [])  # type: ignore[arg-type]
            for ep in eps:
                key = f"{group}:{ep.name}"
                if key in seen_names:
                    continue
                seen_names.add(key)
                out.append(
                    PluginInfo(
                        name=ep.name,
                        source=PluginSource.ENTRY_POINT,
                        location=f"{group}={getattr(ep, 'value', ep.name)}",
                    )
                )
        return out

    def discover(self) -> list[PluginInfo]:
        """Return every plugin found in the three sources, in load order."""
        infos: list[PluginInfo] = []
        seen: set[str] = set()
        disabled = self._disabled_set()

        def _push(plugin: PluginInfo) -> None:
            if plugin.name in seen:
                logger.warning(
                    "Plugin %r already loaded from a higher-priority source; "
                    "ignoring duplicate at %s",
                    plugin.name,
                    plugin.location,
                )
                return
            if plugin.name in disabled:
                plugin.enabled = False
            seen.add(plugin.name)
            infos.append(plugin)

        for plugin in self._discover_directory(self.user_dir, PluginSource.USER):
            _push(plugin)
        if self.project_dir is not None:
            project_plugins_dir = self.project_dir / ".cytopert" / "plugins"
            for plugin in self._discover_directory(
                project_plugins_dir, PluginSource.PROJECT
            ):
                _push(plugin)
        for plugin in self._discover_entry_points():
            _push(plugin)
        return infos

    # ----- Loading ---------------------------------------------------

    def _load_directory_plugin(self, info: PluginInfo) -> None:
        spec = importlib.util.spec_from_file_location(
            f"cytopert_plugin_{info.source.value}_{info.name}",
            info.location,
        )
        if spec is None or spec.loader is None:
            info.error = f"Could not build import spec for {info.location}"
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            info.error = f"Import failed: {exc}"
            return
        setup = getattr(module, "setup", None)
        if not callable(setup):
            info.error = "Plugin module exposes no setup(ctx) callable"
            return
        info.setup = setup

    def _load_entry_point_plugin(self, info: PluginInfo) -> None:
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover
            info.error = "importlib.metadata.entry_points unavailable"
            return
        for group in PLUGIN_ENTRYPOINT_GROUPS:
            try:
                eps = entry_points(group=group)
            except TypeError:
                eps = entry_points().get(group, [])  # type: ignore[arg-type]
            for ep in eps:
                if ep.name != info.name:
                    continue
                try:
                    setup = ep.load()
                except Exception as exc:
                    info.error = f"entry_point load failed: {exc}"
                    return
                if not callable(setup):
                    info.error = "Entry-point object is not callable"
                    return
                info.setup = setup
                return
        info.error = "Entry point disappeared between discovery and load"

    def load(self, infos: list[PluginInfo]) -> None:
        """Resolve every ``info.setup`` for the discovered plugins."""
        for info in infos:
            if not info.enabled:
                continue
            try:
                if info.source is PluginSource.ENTRY_POINT:
                    self._load_entry_point_plugin(info)
                else:
                    self._load_directory_plugin(info)
            except Exception as exc:  # defensive; per-plugin loaders catch most
                info.error = f"unexpected loader error: {exc}"

    # ----- Setup -----------------------------------------------------

    def setup_all(
        self, context_factory: Callable[[PluginInfo], PluginContext]
    ) -> list[PluginInfo]:
        """Discover, load, and call setup() for every enabled plugin."""
        infos = self.discover()
        self.load(infos)
        for info in infos:
            if not info.enabled or info.setup is None:
                continue
            try:
                info.setup(context_factory(info))
            except Exception as exc:
                info.error = f"setup() raised: {exc}"
                logger.warning(
                    "Plugin %r setup failed (%s): %s",
                    info.name,
                    info.location,
                    exc,
                )
        return infos

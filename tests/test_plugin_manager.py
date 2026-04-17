"""Stage 7.3 tests for the plugin discovery / loader."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

from cytopert.agent.tools.registry import ToolRegistry
from cytopert.plugins.manager import (
    DEFAULT_DISABLED_FILE,
    PluginContext,
    PluginManager,
    PluginSource,
)


def _write_plugin(base: Path, name: str, tool_name: str) -> None:
    pdir = base / name
    pdir.mkdir(parents=True, exist_ok=True)
    body = textwrap.dedent(
        f"""
        import json

        async def _handler(**kwargs):
            return json.dumps({{"echo": "{tool_name}"}})

        def setup(ctx):
            ctx.register_tool(
                name="{tool_name}",
                schema={{"type": "object", "properties": {{}}}},
                handler=_handler,
                description="{tool_name} plugin tool",
            )
        """
    )
    (pdir / "cytopert_plugin.py").write_text(body, encoding="utf-8")


def _ctx_factory_for(reg: ToolRegistry, workspace: Path):
    def _factory(info):
        return PluginContext(
            info=info, registry=reg, workspace=workspace,
            evidence_db=None, memory=None, chain_store=None,
        )
    return _factory


def test_discovery_user_and_project_sources(tmp_path: Path) -> None:
    user_dir = tmp_path / "user_plugins"
    project = tmp_path / "project"
    user_dir.mkdir()
    (project / ".cytopert" / "plugins").mkdir(parents=True)
    _write_plugin(user_dir, "u_one", "echo_user")
    _write_plugin(project / ".cytopert" / "plugins", "p_one", "echo_project")

    mgr = PluginManager(user_dir=user_dir, project_dir=project)
    infos = mgr.discover()
    by_source = {(i.name, i.source) for i in infos}
    assert ("u_one", PluginSource.USER) in by_source
    assert ("p_one", PluginSource.PROJECT) in by_source


def test_disabled_txt_marks_plugin_as_disabled(tmp_path: Path) -> None:
    user_dir = tmp_path / "user_plugins"
    user_dir.mkdir()
    _write_plugin(user_dir, "u_one", "echo_user")
    (user_dir / DEFAULT_DISABLED_FILE).write_text("u_one\n# comment line\n")
    mgr = PluginManager(user_dir=user_dir)
    info = next(i for i in mgr.discover() if i.name == "u_one")
    assert info.enabled is False


def test_setup_registers_tool_and_skips_disabled(tmp_path: Path) -> None:
    user_dir = tmp_path / "user_plugins"
    project = tmp_path / "project"
    user_dir.mkdir()
    (project / ".cytopert" / "plugins").mkdir(parents=True)
    _write_plugin(user_dir, "user_plug", "echo_user_plug")
    _write_plugin(project / ".cytopert" / "plugins", "proj_plug", "echo_proj_plug")
    (user_dir / DEFAULT_DISABLED_FILE).write_text("user_plug\n")

    reg = ToolRegistry()
    mgr = PluginManager(user_dir=user_dir, project_dir=project)
    results = mgr.setup_all(_ctx_factory_for(reg, project))
    enabled = {i.name for i in results if i.enabled is not False and i.tools_registered}
    assert enabled == {"proj_plug"}
    assert "echo_proj_plug" in reg
    assert "echo_user_plug" not in reg


def test_setup_failure_recorded_on_info(tmp_path: Path) -> None:
    user_dir = tmp_path / "user_plugins"
    user_dir.mkdir()
    bad_dir = user_dir / "broken"
    bad_dir.mkdir()
    (bad_dir / "cytopert_plugin.py").write_text("def setup(ctx): raise RuntimeError('boom')\n")
    reg = ToolRegistry()
    mgr = PluginManager(user_dir=user_dir)
    results = mgr.setup_all(_ctx_factory_for(reg, user_dir))
    info = next(i for i in results if i.name == "broken")
    assert info.error and "boom" in info.error
    assert "broken" not in reg.tool_names


def test_duplicate_name_keeps_higher_priority(tmp_path: Path) -> None:
    user_dir = tmp_path / "user_plugins"
    project = tmp_path / "project"
    user_dir.mkdir()
    (project / ".cytopert" / "plugins").mkdir(parents=True)
    _write_plugin(user_dir, "echo", "echo_user_priority")
    _write_plugin(project / ".cytopert" / "plugins", "echo", "echo_project_priority")
    mgr = PluginManager(user_dir=user_dir, project_dir=project)
    results = mgr.discover()
    names = [i.name for i in results]
    assert names.count("echo") == 1
    info = next(i for i in results if i.name == "echo")
    assert info.source is PluginSource.USER


def test_registered_tool_dispatch(tmp_path: Path) -> None:
    user_dir = tmp_path / "user_plugins"
    user_dir.mkdir()
    _write_plugin(user_dir, "ping_pkg", "ping_pkg")
    reg = ToolRegistry()
    mgr = PluginManager(user_dir=user_dir)
    mgr.setup_all(_ctx_factory_for(reg, user_dir))
    out = asyncio.run(reg.execute("ping_pkg", {}))
    assert "ping_pkg" in out

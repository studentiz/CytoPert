"""CytoPert plugin discovery (stage 7.3).

Adapted in spirit from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:hermes_cli/plugins.py (MIT
License). See docs/hermes-borrowing.md for the per-module diff
rationale.

Public surface:
    * ``PluginManager`` -- discovers and loads tool / scenario plugins.
    * ``PluginContext`` -- handed to plugin ``setup`` callables; exposes
      ``register_tool`` and ``register_scenario`` plus references to the
      live AgentLoop subsystems (workspace, evidence_db, memory).
"""

from cytopert.plugins.manager import (
    DEFAULT_DISABLED_FILE,
    PluginContext,
    PluginInfo,
    PluginManager,
    PluginSource,
)

__all__ = [
    "DEFAULT_DISABLED_FILE",
    "PluginContext",
    "PluginInfo",
    "PluginManager",
    "PluginSource",
]

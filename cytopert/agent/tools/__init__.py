"""Tools available to the CytoPert agent.

Re-exports the public surface used by ``AgentLoop._register_default_tools``
and external test harnesses. Stub tools (pertpy_*, pathway_*, decoupler) and
their re-exports were removed in stage 1 of the completeness overhaul; the
``pathway_lookup`` tool added in stage 7.2 will appear here once it lands.
"""

from cytopert.agent.tools.base import Tool
from cytopert.agent.tools.census import CensusQueryTool, LoadLocalH5adTool
from cytopert.agent.tools.chain_status import ChainStatusTool
from cytopert.agent.tools.chains import ChainsTool
from cytopert.agent.tools.evidence import EvidenceTool
from cytopert.agent.tools.evidence_search import EvidenceSearchTool
from cytopert.agent.tools.registry import (
    ToolEntry,
    ToolRegistry,
    registry,
    tool_error,
    tool_result,
)
from cytopert.agent.tools.scanpy_tools import (
    ScanpyClusterTool,
    ScanpyDETool,
    ScanpyPreprocessTool,
)

__all__ = [
    "Tool",
    "ToolEntry",
    "ToolRegistry",
    "registry",
    "tool_error",
    "tool_result",
    "CensusQueryTool",
    "LoadLocalH5adTool",
    "EvidenceTool",
    "EvidenceSearchTool",
    "ChainsTool",
    "ChainStatusTool",
    "ScanpyClusterTool",
    "ScanpyDETool",
    "ScanpyPreprocessTool",
]

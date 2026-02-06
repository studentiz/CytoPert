"""Agent loop: the core processing engine for CytoPert."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from cytopert.agent.context import ContextBuilder
from cytopert.agent.tools.registry import ToolRegistry
from cytopert.agent.tools.census import CensusQueryTool, LoadLocalH5adTool
from cytopert.agent.tools.evidence import EvidenceTool
from cytopert.agent.tools.scanpy_tools import ScanpyPreprocessTool, ScanpyClusterTool, ScanpyDETool
from cytopert.agent.tools.pertpy_tools import PertpyPerturbationDistanceTool, PertpyDifferentialResponseTool
from cytopert.agent.tools.decoupler_tools import DecouplerEnrichmentTool
from cytopert.agent.tools.pathway import PathwayConstraintTool, PathwayCheckTool
from cytopert.agent.tools.chains import ChainsTool
from cytopert.data.evidence_builder import build_evidence_summary
from cytopert.data.models import EvidenceEntry
from cytopert.providers.base import LLMProvider
from cytopert.session.manager import SessionManager


class AgentLoop:
    """
    Agent loop: receive task -> build context -> call LLM -> execute tools -> output.
    Tools: census_query, load_local_h5ad, evidence; plus scanpy/pertpy/decoupler/pathway/chains when registered.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self._evidence_store: list[EvidenceEntry] = []
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register census, evidence, and scverse tools."""
        self.tools.register(CensusQueryTool())
        self.tools.register(LoadLocalH5adTool())
        self.tools.register(EvidenceTool(evidence_store=self._evidence_store))
        self.tools.register(ScanpyPreprocessTool(self.workspace))
        self.tools.register(ScanpyClusterTool(self.workspace))
        self.tools.register(ScanpyDETool())
        self.tools.register(PertpyPerturbationDistanceTool(self.workspace))
        self.tools.register(PertpyDifferentialResponseTool())
        self.tools.register(DecouplerEnrichmentTool(self.workspace))
        self.tools.register(PathwayConstraintTool())
        self.tools.register(PathwayCheckTool())
        self.tools.register(ChainsTool())

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
    ) -> str:
        """
        Process a message directly (CLI or workflow).
        Returns the agent's text response.
        """
        session = self.sessions.get_or_create(session_key)
        evidence_summary = build_evidence_summary(self._evidence_store) if self._evidence_store else None
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=content,
            evidence_summary=evidence_summary,
        )
        iteration = 0
        final_content = None
        tools_used = False
        tool_results: list[str] = []
        forced = self._maybe_parse_forced_tool_call(content)
        if forced:
            tool_name, params = forced
            tool_defs = self.tools.get_definitions()
            tool_call_dicts = [
                {
                    "id": "forced_tool_call",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(params),
                    },
                }
            ]
            messages = self.context.add_assistant_message(messages, "", tool_call_dicts)
            result = await self.tools.execute(tool_name, params)
            tool_results.append(result)
            tools_used = True
            messages = self.context.add_tool_result(messages, "forced_tool_call", tool_name, result)
            response = await self.provider.chat(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                model=self.model,
            )
            if response.finish_reason == "error" or (
                response.content and response.content.startswith("Error calling LLM")
            ):
                tool_msg = tool_results[-1] if tool_results else "No tool result."
                final_content = (
                    f"工具已执行，结果如下：\n{tool_msg}\n\nLLM 响应失败：{response.content}"
                )
            else:
                final_content = response.content
        else:
            while iteration < self.max_iterations:
                iteration += 1
                tool_defs = self.tools.get_definitions()
                response = await self.provider.chat(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    model=self.model,
                )
                if response.has_tool_calls:
                    tools_used = True
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages = self.context.add_assistant_message(
                        messages, response.content, tool_call_dicts
                    )
                    for tool_call in response.tool_calls:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        tool_results.append(result)
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                else:
                    final_content = response.content
                    break
        if final_content is None:
            final_content = "I completed processing but have no final response."
        if not self._evidence_store and (not tools_used or self._all_tool_results_errors(tool_results)):
            final_content = self._enforce_evidence_gate(final_content, tool_results)
        session.add_message("user", content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        return final_content

    def _enforce_evidence_gate(self, content: str | None, tool_results: list[str] | None = None) -> str:
        """If no evidence/tools were used, require data instead of speculative answers."""
        errors = []
        if tool_results:
            errors = [r for r in tool_results if r.strip().lower().startswith("error")]
        error_msg = ""
        if errors:
            error_msg = "工具调用失败信息：\n- " + "\n- ".join(errors) + "\n\n"
        return (
            f"{error_msg}当前没有任何证据条目或数据来源，无法给出可靠的基因清单或上下调影响结论。\n"
            "请提供以下之一，我再调用工具获取证据：\n"
            "- 本地 h5ad 文件路径（含脓毒症相关样本与注释）\n"
            "- 允许我使用 cellxgene Census，并提供组织/细胞类型/疾病筛选条件\n"
            "如果你愿意，我也可以先调用 `census_query` 做数据探索（请给出过滤条件）。"
        )

    @staticmethod
    def _all_tool_results_errors(results: list[str]) -> bool:
        if not results:
            return True
        return all(r.strip().lower().startswith("error") for r in results)

    def _maybe_parse_forced_tool_call(self, content: str) -> tuple[str, dict[str, Any]] | None:
        """If user explicitly asks for a tool call, parse minimal params and force execution."""
        text = content or ""
        if "census_query" not in text:
            return None
        params: dict[str, Any] = {}
        obs = self._extract_filter(text, "obs_value_filter")
        var = self._extract_filter(text, "var_value_filter")
        census_version = self._extract_filter(text, "census_version")
        timeout_seconds = self._extract_timeout_seconds(text)
        organism = self._extract_filter(text, "organism")
        obs_only = self._extract_bool(text, "obs_only")
        obs_coords = self._extract_filter(text, "obs_coords")
        max_cells = self._extract_filter(text, "max_cells")
        if obs:
            params["obs_value_filter"] = obs
        if var:
            params["var_value_filter"] = var
        if census_version:
            params["census_version"] = census_version
        if timeout_seconds:
            params["timeout_seconds"] = int(timeout_seconds)
        if organism:
            params["organism"] = organism
        if obs_only is not None:
            params["obs_only"] = obs_only
        if obs_coords:
            params["obs_coords"] = obs_coords
        if max_cells and max_cells.isdigit():
            params["max_cells"] = int(max_cells)
        return ("census_query", params)

    @staticmethod
    def _extract_timeout_seconds(text: str) -> str | None:
        m = re.search(r"timeout_seconds\s*(?:=|：|:)?\s*(\d+)", text, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1)

    @staticmethod
    def _extract_bool(text: str, key: str) -> bool | None:
        m = re.search(rf"{key}\s*(?:=|：|:)?\s*(true|false|1|0|yes|no)", text, flags=re.IGNORECASE)
        if not m:
            return None
        val = m.group(1).lower()
        return val in {"true", "1", "yes"}

    @staticmethod
    def _extract_filter(text: str, key: str) -> str | None:
        pattern = rf"{key}\s*(?:=|：|:|设为)?\s*([^\n，。]+)"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return None
        value = m.group(1).strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        value = re.sub(r"（[^）]*）$", "", value).strip()
        value = re.sub(r"\([^\)]*\)$", "", value).strip()
        return value

    @staticmethod
    def _looks_like_data_request(text: str) -> bool:
        """Heuristic to see if the model already asked for data."""
        if not text:
            return False
        keywords = [
            "请提供", "需要", "数据", "h5ad", "cellxgene", "census", "过滤",
            "允许我", "权限", "dataset", "数据集"
        ]
        return any(k in text for k in keywords)

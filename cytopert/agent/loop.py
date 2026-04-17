"""Agent loop: the core processing engine for CytoPert.

Hermes-style learning loop additions (see [docs/overview.md] for the bigger picture):

- Persistent EvidenceDB (SQLite + FTS5) replaces the per-process evidence_store as the
  long-term episodic memory. Tool calls that produce data evidence are mirrored into the
  DB on every turn.
- MemoryStore renders a frozen snapshot into the system prompt at the start of each
  ``process_direct`` invocation; the in-session ``memory`` tool persists changes to disk
  but does not rewrite the live messages list (mirrors Hermes' prompt-cache discipline).
- SkillsManager exposes a Level-0 index in the system prompt; full SKILL.md bodies are
  fetched on demand via ``skill_view``.
- ChainStore records every MechanismChain with a status state machine.
- A reflection hook (``cytopert.agent.reflection``) runs at the end of complex turns and
  may stage skills / write memory / update chain status.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cytopert.agent.context import ContextBuilder
from cytopert.agent.tools.census import CensusQueryTool, LoadLocalH5adTool
from cytopert.agent.tools.chain_status import ChainStatusTool
from cytopert.agent.tools.chains import ChainsTool
from cytopert.agent.tools.decoupler_tools import DecouplerEnrichmentTool
from cytopert.agent.tools.evidence import EvidenceTool
from cytopert.agent.tools.evidence_search import EvidenceSearchTool
from cytopert.agent.tools.pathway import PathwayCheckTool, PathwayConstraintTool
from cytopert.agent.tools.pertpy_tools import (
    PertpyDifferentialResponseTool,
    PertpyPerturbationDistanceTool,
)
from cytopert.agent.tools.registry import ToolRegistry
from cytopert.agent.tools.scanpy_tools import ScanpyClusterTool, ScanpyDETool, ScanpyPreprocessTool
from cytopert.data.evidence_builder import build_evidence_summary, record_tool_evidence
from cytopert.data.models import EvidenceEntry
from cytopert.memory.store import MemoryStore
from cytopert.memory.tool import MemoryTool
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.evidence_db import EvidenceDB
from cytopert.providers.base import LLMProvider
from cytopert.session.manager import SessionManager
from cytopert.skills.manager import SkillsManager
from cytopert.skills.tool import SkillManageTool, SkillsListTool, SkillViewTool
from cytopert.utils.helpers import get_chains_dir, get_memory_dir, get_skills_dir, get_state_db_path

_REFLECTION_TRIGGERS = {
    "min_tool_calls": 5,
    "min_evidence_entries": 3,
}


class AgentLoop:
    """Receive task -> build context -> call LLM -> execute tools -> reflect -> output."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        memory_store: MemoryStore | None = None,
        skills_manager: SkillsManager | None = None,
        evidence_db: EvidenceDB | None = None,
        chain_store: ChainStore | None = None,
        enable_reflection: bool = True,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)

        self.memory = memory_store or MemoryStore(get_memory_dir())
        self.skills = skills_manager or SkillsManager(get_skills_dir())
        self.evidence_db = evidence_db or EvidenceDB(get_state_db_path())
        self.chains = chain_store or ChainStore(get_state_db_path(), get_chains_dir())
        self.enable_reflection = enable_reflection

        try:
            self.skills.install_bundled()
        except Exception:
            pass

        self.tools = ToolRegistry()
        self._evidence_store: list[EvidenceEntry] = []
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register census, evidence, scverse, memory, skills, evidence_search, chain_status."""
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
        self.tools.register(ChainsTool(store=self.chains))
        self.tools.register(MemoryTool(self.memory))
        self.tools.register(SkillsListTool(self.skills))
        self.tools.register(SkillViewTool(self.skills))
        self.tools.register(SkillManageTool(self.skills))
        self.tools.register(EvidenceSearchTool(self.evidence_db))
        self.tools.register(ChainStatusTool(self.chains))

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
    ) -> str:
        """Process a message directly (CLI or workflow). Returns the agent's text response."""
        session = self.sessions.get_or_create(session_key)
        evidence_summary = (
            build_evidence_summary(self._evidence_store) if self._evidence_store else None
        )
        memory_snapshot = self.memory.render_snapshot()
        available_tools = set(self.tools.tool_names)
        skills_index = self.skills.render_index(available_tools=available_tools)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=content,
            evidence_summary=evidence_summary,
            memory_snapshot=memory_snapshot,
            skills_index=skills_index,
        )
        iteration = 0
        final_content = None
        tools_used = False
        tool_calls_count = 0
        chains_touched: list[str] = []
        new_evidence_ids: list[str] = []
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
            tool_calls_count += 1
            self._record_side_effects(tool_name, params, result, session_key,
                                      chains_touched, new_evidence_ids)
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
                        tool_calls_count += 1
                        self._record_side_effects(tool_call.name, tool_call.arguments, result,
                                                  session_key, chains_touched, new_evidence_ids)
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                else:
                    final_content = response.content
                    break

        if final_content is None:
            final_content = "I completed processing but have no final response."
        if not self._evidence_store and (
            not tools_used or self._all_tool_results_errors(tool_results)
        ):
            final_content = self._enforce_evidence_gate(final_content, tool_results)

        session.add_message("user", content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        if self.enable_reflection:
            try:
                await self._maybe_reflect(
                    session_key=session_key,
                    user_message=content,
                    final_response=final_content,
                    tool_calls_count=tool_calls_count,
                    chains_touched=chains_touched,
                    new_evidence_ids=new_evidence_ids,
                )
            except Exception:
                pass

        return final_content

    def _record_side_effects(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: str,
        session_key: str,
        chains_touched: list[str],
        new_evidence_ids: list[str],
    ) -> None:
        """Persist evidence entries and track chain ids touched during this turn."""
        entry = record_tool_evidence(tool_name, params, result, session_id=session_key)
        if entry is not None:
            self._evidence_store.append(entry)
            try:
                self.evidence_db.add(entry, session_id=session_key)
                new_evidence_ids.append(entry.id)
            except Exception:
                pass

        if tool_name in {"chains", "chain_status"}:
            try:
                payload = json.loads(result)
                cid = payload.get("chain_id")
                if cid:
                    chains_touched.append(cid)
            except (json.JSONDecodeError, TypeError):
                pass

    async def _maybe_reflect(
        self,
        session_key: str,
        user_message: str,
        final_response: str,
        tool_calls_count: int,
        chains_touched: list[str],
        new_evidence_ids: list[str],
    ) -> None:
        from cytopert.agent.reflection import maybe_reflect

        await maybe_reflect(
            loop=self,
            session_key=session_key,
            user_message=user_message,
            final_response=final_response,
            tool_calls_count=tool_calls_count,
            chains_touched=chains_touched,
            new_evidence_ids=new_evidence_ids,
            triggers=_REFLECTION_TRIGGERS,
        )

    def _enforce_evidence_gate(
        self, content: str | None, tool_results: list[str] | None = None
    ) -> str:
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

    _FORCED_PARSE_KEYS = (
        "obs_value_filter",
        "var_value_filter",
        "census_version",
        "organism",
        "obs_only",
        "obs_columns",
        "obs_coords",
        "max_cells",
        "timeout_seconds",
    )

    @classmethod
    def _extract_filter(cls, text: str, key: str) -> str | None:
        # Stop at the next known parameter key, an English/Chinese comma, newline,
        # or a Chinese full-stop. This prevents `obs_value_filter=tissue == 'blood',
        # obs_only=true, max_cells=200` from being captured as one giant filter.
        other = "|".join(re.escape(k) for k in cls._FORCED_PARSE_KEYS if k != key)
        stop = rf"(?:,|，|\n|。|\s+(?:{other})\b|$)"
        pattern = rf"{key}\s*(?:=|：|:|设为)?\s*(.+?)\s*{stop}"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return None
        value = m.group(1).strip().rstrip(",，。")
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        value = re.sub(r"（[^）]*）$", "", value).strip()
        value = re.sub(r"\([^\)]*\)$", "", value).strip()
        return value or None

    @staticmethod
    def _looks_like_data_request(text: str) -> bool:
        """Heuristic to see if the model already asked for data."""
        if not text:
            return False
        keywords = [
            "请提供", "需要", "数据", "h5ad", "cellxgene", "census", "过滤",
            "允许我", "权限", "dataset", "数据集",
        ]
        return any(k in text for k in keywords)

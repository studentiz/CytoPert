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
import logging
import re
from pathlib import Path
from typing import Any

from cytopert.agent.context import ContextBuilder
from cytopert.agent.context_compressor import CytoPertCompressor
from cytopert.agent.context_engine import ContextEngine
from cytopert.agent.tools.census import CensusQueryTool, LoadLocalH5adTool
from cytopert.agent.tools.chain_status import ChainStatusTool
from cytopert.agent.tools.chains import ChainsTool
from cytopert.agent.tools.evidence import EvidenceTool
from cytopert.agent.tools.evidence_search import EvidenceSearchTool
from cytopert.agent.tools.pathway_lookup import PathwayLookupTool
from cytopert.agent.tools.registry import ToolRegistry
from cytopert.agent.tools.scanpy_tools import (
    ScanpyClusterTool,
    ScanpyDETool,
    ScanpyPreprocessTool,
)
from cytopert.data.evidence_builder import build_evidence_summary, record_tool_evidence
from cytopert.data.models import EvidenceEntry
from cytopert.memory.store import MemoryStore
from cytopert.memory.tool import MemoryTool
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.evidence_db import EvidenceDB
from cytopert.plugins.manager import PluginContext, PluginInfo, PluginManager
from cytopert.providers.base import LLMProvider
from cytopert.session.manager import SessionManager
from cytopert.skills.manager import SkillsManager
from cytopert.skills.tool import SkillManageTool, SkillsListTool, SkillViewTool
from cytopert.utils.helpers import get_chains_dir, get_memory_dir, get_skills_dir, get_state_db_path

logger = logging.getLogger(__name__)

_REFLECTION_TRIGGERS = {
    "min_tool_calls": 5,
    "min_evidence_entries": 3,
}

# Backfill the in-process evidence_store with the most recent N entries from
# the persistent EvidenceDB. Without this, a fresh AgentLoop would render an
# empty Evidence Summary into the prompt even though a previous session had
# produced data.
_EVIDENCE_BACKFILL_LIMIT = 20

# Keywords that mark the user's message as a "research conclusion" question
# -- the only kind that should trigger the evidence gate. Keep the list
# small and unambiguous; substring matching is intentional so e.g.
# "differential expression" matches "differential".
_RESEARCH_KEYWORDS = (
    "gene list", "top genes", "differentially", "differential",
    "rank genes", "pathway", "enrichment", "upregulated", "downregulated",
    "up-regulated", "down-regulated", "fold change", "logfc", "lfc",
    "p-value", "adjusted p", "padj", "fdr", "DE genes", "deg",
    "perturbation distance", "trajectory", "cluster markers",
)

# Plan-before-execute state machine.
#
# Interactive sessions (``cytopert agent`` without ``-m``) start in
# ``awaiting_plan``; the agent is told via the system prompt to output a
# textual plan only and any ``tool_calls`` returned in the first turn are
# discarded. The user moves the session to ``executing`` by typing one of
# ``GO_PHRASES`` (case-insensitive). One-shot ``-m`` invocations skip the
# gate entirely because there is no second turn to authorise execution.
PLAN_MODE_KEY = "plan_mode"
PLAN_MODE_AWAITING = "awaiting_plan"
PLAN_MODE_EXECUTING = "executing"
PLAN_MODE_DISABLED = "disabled"

GO_PHRASES = (
    "go", "go!", "go.", "execute", "run it", "run", "approve", "approved",
    "confirm", "confirmed", "proceed", "yes", "yep", "ok", "okay", "lgtm",
)


def _is_go_phrase(message: str) -> bool:
    """Return True iff *message* (after stripping) is a plain go-signal."""
    if not message:
        return False
    cleaned = message.strip().rstrip(".!?,").lower()
    return cleaned in GO_PHRASES


_PLAN_GATE_INSTRUCTION = (
    "## Plan Gate (active)\n"
    "This session is in PLAN mode. For THIS turn only, output a numbered "
    "execution plan in plain text. Do NOT call any tools yet -- wait for "
    "the researcher to type 'go' (or 'execute' / 'approve') in the next "
    "turn before invoking any tool. Tool calls emitted during this turn "
    "will be discarded."
)

# Evidence binding enforcement.
#
# Any final reply that cites evidence MUST do so with the form
# ``[evidence: id_a, id_b]`` or ``(evidence: id_c)``. The enforcer parses
# all such citations out of the reply, looks each id up in the
# EvidenceDB / in-process store, and -- if any id is missing -- runs a
# single retry turn asking the model to fix its references using only
# real ids. If the second attempt still mentions phantom ids, we leave
# them in place but append a 'The following evidence ids were not found'
# advisory so the user can react.
_EVIDENCE_REF_RE = re.compile(
    r"\[evidence:\s*([^\]]+)\]|\(evidence:\s*([^)]+)\)",
    flags=re.IGNORECASE,
)


def _split_ids(raw: str) -> list[str]:
    """Split a comma-separated id list and strip surrounding whitespace / quotes."""
    out: list[str] = []
    for token in raw.split(","):
        cleaned = token.strip().strip("'\"")
        if cleaned:
            out.append(cleaned)
    return out


def _extract_evidence_citations(text: str | None) -> list[str]:
    """Return every evidence id mentioned via ``[evidence: ...]`` / ``(evidence: ...)``."""
    if not text:
        return []
    found: list[str] = []
    for bracket, paren in _EVIDENCE_REF_RE.findall(text):
        for eid in _split_ids(bracket or paren):
            if eid not in found:
                found.append(eid)
    return found


_EVIDENCE_BINDING_PROMPT = (
    "## Evidence Binding (CytoPert hard constraint)\n"
    "When you cite evidence in your final reply, use the form "
    "`[evidence: id_a, id_b]` or `(evidence: id_c)`. "
    "Every id you cite MUST exist in the evidence store -- the IDs "
    "produced by tool calls follow the `tool_<tool_name>_<digest>` "
    "shape (e.g. `tool_scanpy_de_3a4b5c6d7e`). Never invent an id; if "
    "you do not have evidence for a claim, omit the citation rather "
    "than fabricate one."
)


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
        max_tokens: int = 8192,
        temperature: float = 0.3,
        context_engine: ContextEngine | None = None,
        save_trajectory: bool = False,
        plugin_manager: PluginManager | None = None,
        load_plugins: bool = True,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        # These mirror the AgentDefaults config so config.json values for
        # temperature / max_tokens actually reach provider.chat. The legacy
        # call path passed neither and silently used LiteLLM's defaults
        # (max_tokens=4096, temperature=0.7).
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)

        self.memory = memory_store or MemoryStore(get_memory_dir())
        self.skills = skills_manager or SkillsManager(get_skills_dir())
        self.evidence_db = evidence_db or EvidenceDB(get_state_db_path())
        self.chains = chain_store or ChainStore(get_state_db_path(), get_chains_dir())
        self.enable_reflection = enable_reflection

        # Default ContextEngine summarises the middle of long conversations
        # with the same provider/model as the main loop. Callers can swap
        # in their own engine (or pass None to disable compression) via
        # the ``context_engine`` constructor kwarg or the
        # ``set_context_engine`` setter.
        self.context_engine: ContextEngine | None = context_engine or CytoPertCompressor(
            provider=self.provider, model=self.model
        )
        # Trajectory recording is opt-in; the CLI exposes this via
        # `cytopert agent --save-trajectory` so casual sessions do not
        # quietly accumulate JSONL files in ~/.cytopert/trajectories/.
        self.save_trajectory = bool(save_trajectory)

        try:
            self.skills.install_bundled()
        except Exception as exc:
            logger.warning("install_bundled failed: %s", exc)

        self.tools = ToolRegistry()
        self._evidence_store: list[EvidenceEntry] = []
        self._backfill_evidence_store()
        self._register_default_tools()
        self.plugin_manager: PluginManager | None = (
            plugin_manager if plugin_manager is not None else (
                PluginManager(project_dir=Path.cwd()) if load_plugins else None
            )
        )
        self.plugins: list[PluginInfo] = []
        if self.plugin_manager is not None:
            try:
                self.plugins = self.plugin_manager.setup_all(self._build_plugin_ctx)
            except Exception as exc:
                logger.warning("plugin setup_all failed: %s", exc)

    @staticmethod
    def _record_usage(session: Any, response: Any) -> None:
        """Accumulate prompt / completion / cost into session.metadata.

        Stored as a dict so a future ``cytopert sessions show`` (or the
        existing ``cytopert status`` view) can break down spend by
        session without re-reading the JSONL transcript. We never raise
        from this helper -- bookkeeping must not crash a turn.
        """
        usage = getattr(response, "usage", None) or {}
        cost_usd = getattr(response, "cost_usd", None)
        if not usage and cost_usd is None:
            return
        try:
            stats = session.metadata.setdefault(
                "usage",
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            )
            stats["calls"] = int(stats.get("calls", 0)) + 1
            stats["prompt_tokens"] = int(stats.get("prompt_tokens", 0)) + int(
                usage.get("prompt_tokens", 0) or 0
            )
            stats["completion_tokens"] = int(stats.get("completion_tokens", 0)) + int(
                usage.get("completion_tokens", 0) or 0
            )
            if cost_usd is not None:
                stats["cost_usd"] = float(stats.get("cost_usd", 0.0)) + float(cost_usd)
        except Exception as exc:
            logger.debug("usage accounting failed: %s", exc)

    def _build_plugin_ctx(self, info: PluginInfo) -> PluginContext:
        """Construct the PluginContext handed to a plugin setup() call."""
        return PluginContext(
            info=info,
            registry=self.tools,
            workspace=self.workspace,
            evidence_db=self.evidence_db,
            memory=self.memory,
            chain_store=self.chains,
        )

    def _backfill_evidence_store(self) -> None:
        """Pre-populate the in-process evidence store from the persistent DB.

        ``build_evidence_summary`` reads exclusively from this list, so without
        backfill a brand-new process would always render an empty Evidence
        Summary even when EvidenceDB on disk has thousands of entries.
        """
        try:
            recent = self.evidence_db.recent(limit=_EVIDENCE_BACKFILL_LIMIT)
        except Exception as exc:
            logger.warning("evidence_db.recent backfill failed: %s", exc)
            return
        # ``recent`` returns most-recent-first; reverse so the in-process
        # store remains chronologically ordered for the summary tail slicing.
        self._evidence_store.extend(reversed(recent))

    def _register_default_tools(self) -> None:
        """Wire built-in CytoPert tools into the registry.

        Stub tools removed in stage 1: pertpy_perturbation_distance,
        pertpy_differential_response, decoupler_enrichment, pathway_check,
        pathway_constraint. Those tools advertised analysis capabilities the
        underlying handlers never delivered. See docs/hermes-borrowing.md
        and the stage 1 commit for context. The pathway_lookup tool added
        in stage 7.2 will replace the pathway_* surface.
        """
        self.tools.register(CensusQueryTool())
        self.tools.register(LoadLocalH5adTool())
        self.tools.register(EvidenceTool(evidence_store=self._evidence_store))
        self.tools.register(ScanpyPreprocessTool(self.workspace))
        self.tools.register(ScanpyClusterTool(self.workspace))
        self.tools.register(ScanpyDETool())
        self.tools.register(PathwayLookupTool())
        self.tools.register(ChainsTool(store=self.chains))
        self.tools.register(MemoryTool(self.memory))
        self.tools.register(SkillsListTool(self.skills))
        self.tools.register(SkillViewTool(self.skills))
        self.tools.register(SkillManageTool(self.skills))
        self.tools.register(EvidenceSearchTool(self.evidence_db))
        # Pass the MemoryStore so chain_status auto-appends a one-line
        # summary to HYPOTHESIS_LOG.md on every transition.
        self.tools.register(ChainStatusTool(self.chains, memory=self.memory))

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        user_feedback: str | None = None,
    ) -> str:
        """Process a message directly (CLI or workflow). Returns the agent's text response.

        ``user_feedback`` carries a structured wet-lab-feedback string from
        ``cytopert agent --feedback`` / ``cytopert run-workflow --feedback``.
        It is forwarded verbatim to the reflection turn so the reflection
        LLM can decide whether to update memory or transition a chain.
        """
        session = self.sessions.get_or_create(session_key)
        if self.context_engine is not None:
            # First time we see this session in this process -- let the
            # engine seed any per-session counters / DB connections.
            try:
                self.context_engine.on_session_start(session_key)
            except Exception as exc:
                logger.debug("context_engine.on_session_start failed: %s", exc)

        # Plan-Gate: decide whether tool calls are allowed in this turn.
        plan_state = session.metadata.get(PLAN_MODE_KEY, PLAN_MODE_DISABLED)
        if plan_state == PLAN_MODE_AWAITING and _is_go_phrase(content):
            session.metadata[PLAN_MODE_KEY] = PLAN_MODE_EXECUTING
            plan_state = PLAN_MODE_EXECUTING
        plan_active = plan_state == PLAN_MODE_AWAITING

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
        if plan_active:
            # Append the gate instruction as an extra system message so the
            # cached system prefix is preserved (LLM prefix caching is the
            # main reason ContextBuilder freezes the system prompt).
            messages.append({"role": "system", "content": _PLAN_GATE_INSTRUCTION})
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
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            # Forced-tool-call branch reuses one chat call; record its
            # usage so spend tracking stays consistent across paths.
            self._record_usage(session, response)
            if response.finish_reason == "error" or (
                response.content and response.content.startswith("Error calling LLM")
            ):
                tool_msg = tool_results[-1] if tool_results else "No tool result."
                final_content = (
                    "Tool executed; the LLM follow-up call failed.\n"
                    f"Tool result:\n{tool_msg}\n\nLLM error: {response.content}"
                )
            else:
                final_content = response.content
        else:
            while iteration < self.max_iterations:
                iteration += 1
                # Pre-call compaction: ask the engine whether to run a
                # post-call compression decision after the next response.
                # We always offer the engine a peek at the messages before
                # the call so a future engine can do a cheap estimate.
                if self.context_engine is not None:
                    try:
                        if self.context_engine.should_compress_preflight(messages):
                            messages = self.context_engine.compress(messages)
                    except Exception as exc:
                        logger.warning(
                            "context_engine pre-flight compress failed: %s", exc
                        )
                # Plan gate: hide the tool catalog so even if the model
                # ignores the system instruction it has nothing to call.
                tool_defs = (
                    [] if plan_active else self.tools.get_definitions()
                )
                response = await self.provider.chat(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                # Accumulate usage / cost into session metadata so the
                # CLI ``status`` command (and downstream observability)
                # can show per-session spend instead of black-boxing it.
                self._record_usage(session, response)
                if self.context_engine is not None and response.usage:
                    try:
                        self.context_engine.update_from_response(response.usage)
                        if self.context_engine.should_compress():
                            messages = self.context_engine.compress(messages)
                    except Exception as exc:
                        logger.warning("context_engine post-call step failed: %s", exc)
                if plan_active and response.has_tool_calls:
                    # Discard tool calls emitted during plan turn; surface
                    # the plan text the model produced as the final reply.
                    final_content = (
                        (response.content or "").strip()
                        + "\n\n[PlanGate] Tool calls were suppressed; "
                        "reply 'go' (or 'execute' / 'approve') to authorise execution."
                    )
                    break
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
            # max_iterations exhausted while the model kept asking for tools.
            # Surface the truncation explicitly instead of returning a generic
            # "no final response" string -- callers can distinguish "model
            # finished" from "loop hit the iteration cap".
            tail = tool_results[-1] if tool_results else "(no tool output captured)"
            final_content = (
                f"Reached the agent loop iteration limit ({self.max_iterations}) "
                f"with no terminal model response. The most recent tool result was:\n\n{tail}\n\n"
                "Re-run with a larger maxToolIterations or a more specific request."
            )
        if (
            not self._evidence_store
            and (not tools_used or self._all_tool_results_errors(tool_results))
            and self._is_research_conclusion(content)
        ):
            # Append the data-request hint instead of replacing the model's
            # reply: chitchat / meta questions get the model's actual answer,
            # research questions get the model's answer plus the hint that
            # without evidence the answer is not reproducible.
            final_content = self._append_evidence_gate(final_content, tool_results)

        # Evidence binding enforcement runs after every successful turn.
        # We always look the cited ids up in the live store so even
        # zero-tool-call turns (where the model relied on the prompt's
        # Evidence Summary) get a fact-check; the retry only fires when
        # the model cited a non-existent id.
        if final_content and not plan_active:
            final_content = await self._enforce_evidence_binding(
                final_content=final_content,
                messages=messages,
                tools_enabled=not plan_active,
            )

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
                    user_feedback=user_feedback,
                )
            except Exception as exc:
                logger.warning("reflection failed: %s", exc)

        if self.save_trajectory:
            try:
                from cytopert.agent.trajectory import (
                    convert_session_to_sharegpt,
                    save_trajectory,
                )

                save_trajectory(
                    convert_session_to_sharegpt(session),
                    model=self.model,
                    completed=True,
                    evidence_ids=new_evidence_ids,
                    chains_touched=chains_touched,
                    session_key=session_key,
                )
            except Exception as exc:
                logger.warning("trajectory save failed: %s", exc)

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
            except Exception as exc:
                # The in-memory store still has the entry, but persistence
                # failed -- log so the divergence is visible instead of
                # silently dropping the row.
                logger.warning(
                    "evidence_db.add failed for %s (%s): %s", entry.id, tool_name, exc
                )
            # Tell the context engine to keep the tool-result message
            # carrying this evidence id verbatim through any future
            # compression -- otherwise downstream chains that cite the
            # id would lose their primary source.
            if self.context_engine is not None:
                try:
                    self.context_engine.protect_evidence(entry.id)
                except Exception as exc:
                    logger.debug("context_engine.protect_evidence failed: %s", exc)

        if tool_name in {"chains", "chain_status"}:
            try:
                payload = json.loads(result)
                cid = payload.get("chain_id")
                if cid:
                    chains_touched.append(cid)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug(
                    "Could not parse chains/chain_status JSON to record chain_id: %s", exc
                )

    async def _maybe_reflect(
        self,
        session_key: str,
        user_message: str,
        final_response: str,
        tool_calls_count: int,
        chains_touched: list[str],
        new_evidence_ids: list[str],
        user_feedback: str | None = None,
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
            user_feedback=user_feedback,
            triggers=_REFLECTION_TRIGGERS,
        )

    def _append_evidence_gate(
        self, content: str | None, tool_results: list[str] | None = None
    ) -> str:
        """Append a data-request hint to the model reply when evidence is empty.

        Only fires when ``_is_research_conclusion`` matched the user message
        AND no evidence was produced (or every tool returned an error). This
        is appended -- not substituted -- so the model's actual answer is
        preserved verbatim and the user only sees an extra advisory block.
        """
        errors: list[str] = []
        if tool_results:
            errors = [r for r in tool_results if r.strip().lower().startswith("error")]
        head = (content or "").rstrip()
        suffix_parts: list[str] = []
        if errors:
            suffix_parts.append(
                "Tool errors during this turn:\n- " + "\n- ".join(errors)
            )
        suffix_parts.append(
            "No evidence entries are available yet, so any reproducible "
            "gene list / pathway / up-or-down conclusion still needs data. "
            "To produce real evidence you can:\n"
            "  - share a local .h5ad path (with the relevant sample annotations),\n"
            "  - allow me to call `census_query` with a tissue / cell-type / disease filter, or\n"
            "  - point me at an existing evidence id via `evidence_search`."
        )
        suffix = "\n\n---\n" + "\n\n".join(suffix_parts)
        return f"{head}{suffix}" if head else suffix.lstrip("\n-")

    def _known_evidence_ids(self) -> set[str]:
        """Set of evidence ids the agent is allowed to cite.

        Pulls from both the in-process store (covers entries created in
        the current turn before they have been persisted) and the
        EvidenceDB recent slice (covers cross-session ids the prompt
        backfill exposed). The DB read failure is logged and ignored so
        the validator falls back to the in-process view when the database
        is unavailable.
        """
        ids = {e.id for e in self._evidence_store if e and e.id}
        try:
            for entry in self.evidence_db.recent(limit=200):
                if entry and entry.id:
                    ids.add(entry.id)
        except Exception as exc:
            logger.debug("evidence_db.recent unavailable for binding check: %s", exc)
        return ids

    async def _enforce_evidence_binding(
        self,
        *,
        final_content: str,
        messages: list[dict[str, Any]],
        tools_enabled: bool,
    ) -> str:
        """Check that every ``[evidence: id]`` citation resolves to a real id.

        On the first miss, run a single follow-up ``provider.chat`` asking
        the model to fix the references using only real ids. If the
        retry still cites phantom ids, leave them in place but append a
        plain-text advisory listing the missing ids so the user (and the
        downstream reflection turn) can react.
        """
        cited = _extract_evidence_citations(final_content)
        if not cited:
            return final_content
        known = self._known_evidence_ids()
        missing = [eid for eid in cited if eid not in known]
        if not missing:
            return final_content

        # Prepare a retry message that lists the offending ids and the
        # ones the model could legitimately cite (capped to 30 to keep
        # the prompt small).
        usable_preview = list(known)[:30]
        retry_user = (
            "Your previous reply cited evidence ids that do not exist in the "
            "evidence store: "
            + ", ".join(missing)
            + ".\n\nValid evidence ids you may cite include (truncated to 30): "
            + (", ".join(usable_preview) if usable_preview else "(none)")
            + ".\n\nPlease repeat the reply with the offending citations either "
            "removed or replaced by real ids. Do not invent new ids."
        )
        retry_messages = list(messages) + [
            {"role": "assistant", "content": final_content},
            {"role": "user", "content": retry_user},
        ]
        try:
            response = await self.provider.chat(
                messages=retry_messages,
                tools=None,  # binding-fix turn must not call tools
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            logger.warning("evidence binding retry call failed: %s", exc)
            response = None

        retried_content = (
            response.content if response and response.content else None
        )
        if retried_content:
            still_missing = [
                eid
                for eid in _extract_evidence_citations(retried_content)
                if eid not in known
            ]
            if not still_missing:
                return retried_content
            # Retry still wrong; surface the diagnostic to the user.
            advisory = (
                "\n\n---\n[Evidence binding] The following cited ids do not "
                "exist in the evidence store after one retry: "
                + ", ".join(still_missing)
                + ". Treat any conclusions that depend on them as unverified."
            )
            return retried_content + advisory
        # Retry failed entirely; keep the original reply but add advisory.
        advisory = (
            "\n\n---\n[Evidence binding] The following cited ids could not be "
            "found in the evidence store: "
            + ", ".join(missing)
            + ". Treat any conclusions that depend on them as unverified."
        )
        return final_content + advisory

    def enable_plan_gate(self, session_key: str) -> None:
        """Mark *session_key* as 'awaiting_plan' so the next turn is plan-only.

        Called by the interactive CLI on session start. ``-m`` (one-shot)
        callers should not use this because there is no follow-up turn to
        carry the ``go`` signal.
        """
        session = self.sessions.get_or_create(session_key)
        if session.metadata.get(PLAN_MODE_KEY) != PLAN_MODE_EXECUTING:
            session.metadata[PLAN_MODE_KEY] = PLAN_MODE_AWAITING
            self.sessions.save(session)

    def reset_plan_gate(self, session_key: str) -> None:
        """Re-arm the plan gate after a ``/reset`` so the next turn plans again."""
        session = self.sessions.get_or_create(session_key)
        session.metadata[PLAN_MODE_KEY] = PLAN_MODE_AWAITING
        self.sessions.save(session)

    @staticmethod
    def _is_research_conclusion(message: str) -> bool:
        """Return True iff the user is asking for a reproducible scientific result.

        Greetings / capability questions / help text must not trigger the
        evidence gate. We use a small substring whitelist of bioinformatics
        terms; this is intentionally permissive on the no-side because
        appending the data-request hint to a chitchat reply is jarring.
        """
        if not message:
            return False
        low = message.lower()
        return any(k.lower() in low for k in _RESEARCH_KEYWORDS)

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


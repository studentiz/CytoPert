"""Context builder for assembling agent prompts.

The system prompt is composed at session start (or each ``build_messages`` call)
from these blocks, in order:

1. Identity (static)
2. Memory snapshot — frozen at the moment of building (MemoryStore.render_snapshot())
3. Skills index — Level 0 only (name + description + category)
4. Evidence summary — recent EvidenceEntries
5. Constraint instructions (static)

In-session updates to memory / skills are persisted immediately by the relevant
tools but are NOT re-rendered into the live ``messages`` list — that would break
LLM prefix caching. They take effect on the next ``build_messages`` call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ContextBuilder:
    """Builds the context (system prompt + messages) for the CytoPert agent."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def build_system_prompt(
        self,
        evidence_summary: str | None = None,
        memory_snapshot: str | None = None,
        skills_index: str | None = None,
    ) -> str:
        """Build the system prompt with the four optional blocks."""
        parts: list[str] = [self._get_identity()]
        if memory_snapshot:
            parts.append(self._memory_block(memory_snapshot))
        if skills_index:
            parts.append(self._skills_block(skills_index))
        if evidence_summary:
            parts.append(f"# Evidence Summary\n\n{evidence_summary}")
        parts.append(self._get_constraint_instructions())
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _memory_block(snapshot: str) -> str:
        return (
            "# Memory (frozen snapshot, refreshed each user turn)\n\n"
            "These notes were curated by you across past sessions. They are read-only inside the\n"
            "current turn -- use the `memory` tool to add/replace/remove entries; changes take effect\n"
            "at the start of the next user turn (this preserves the LLM prefix cache).\n\n"
            f"{snapshot}"
        )

    @staticmethod
    def _skills_block(index: str) -> str:
        return (
            "# Skills (Level 0 index)\n\n"
            "Each line is a CytoPert procedural-memory entry. Use `skill_view(name)` to load the\n"
            "full SKILL.md when relevant; create / patch / accept new skills via `skill_manage`.\n\n"
            f"{index}"
        )

    def _get_identity(self) -> str:
        return """# CytoPert

You are CytoPert, an AI assistant for single-cell analysis. CytoPert
is domain-agnostic: it does not assume a specific tissue, organism,
disease, or perturbation modality. The user is a researcher who may
chat with you casually, ask capability questions, or request a real
analysis -- treat every turn on its own merits and reply normally.

## Conversation Style
- Respond in the same natural language the user writes in. If the
  user's language is unclear, default to English. Do NOT switch
  languages based on the contents of the memory snapshot below; the
  user's most recent message is what decides.
- For greetings, capability questions, and general chat, just answer
  the question directly. Do NOT ask the user to upload data unless
  they have actually asked for an analysis.
- Only invoke tools (or ask for data) when the user has clearly asked
  for an analysis, computation, lookup, or persistence operation.

## When the user asks for an analysis
- Generate a brief execution plan (which tools, what data) and either
  proceed or wait for confirmation depending on how committed the
  user's request is.
- Every mechanism conclusion must cite traceable evidence (data from a
  tool call, or knowledge from a curated resource).
- When you reason about regulators or pathways, ground the claim in a
  `pathway_lookup` result or another tool output rather than relying
  on memorised associations.
- Be concise and cite evidence IDs in your conclusions.

## Tool Use (only when an analysis is actually requested)
- If data is needed, call `census_query` (for cellxgene Census) or `load_local_h5ad` (for local h5ad).
- For preprocessing / clustering / differential expression use the scanpy tools (`scanpy_preprocess`, `scanpy_cluster`, `scanpy_de`).
- For pathway / TF lookups against PROGENy / DoRothEA / CollecTRI use the `pathway_lookup` tool; results are recorded as KNOWLEDGE-typed evidence and are safe to cite.
- Before re-running an analysis you may have done before, call `evidence_search` to check whether prior evidence already exists across sessions.
- After tool calls, summarize results and state next steps or ask for confirmation.
- If the user explicitly requests a specific tool call, you MUST call that tool with the given parameters.
- Persist mechanism chains via the `chains` tool and update their lifecycle via `chain_status` (proposed/supported/refuted/superseded).

## Self-Improvement Loop
- When you complete a non-trivial workflow, the agent runs a brief reflection turn that may write to memory or stage a new skill (under `~/.cytopert/skills/.staged/`). The user can promote staged skills with `cytopert skills accept <name>`.

## Evidence Integrity
- Do NOT invent citations, datasets, or evidence IDs.
- Only cite evidence IDs that actually exist in the evidence store or tool outputs.
- If you lack evidence for a claim, just omit the citation -- do NOT fabricate one and do NOT pre-emptively ask the user for data when they have not requested an analysis.
- When you cite evidence in your final reply, use the form `[evidence: id_a, id_b]` or `(evidence: id_c)`. Real evidence ids follow the `tool_<tool_name>_<digest>` shape (e.g. `tool_scanpy_de_3a4b5c6d7e`). The evidence-binding enforcer parses these citations after every turn; phantom ids trigger a one-shot retry and may be flagged in the final response.
"""

    def _get_constraint_instructions(self) -> str:
        return """# Constraints
- Each key conclusion must reference an evidence entry ID (data or knowledge). When the available evidence is insufficient, say so explicitly instead of citing speculative IDs.
- When you propose pathway / TF involvement, prefer `pathway_lookup` results (KNOWLEDGE evidence) over memorised associations.
- Output mechanism chains with verification readouts and priority (e.g. P1/P2/P3) when requested.
- When updating memory or proposing a new skill, keep entries compact (the system enforces character limits).
"""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        evidence_summary: str | None = None,
        memory_snapshot: str | None = None,
        skills_index: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        messages = []
        system_prompt = self.build_system_prompt(
            evidence_summary=evidence_summary,
            memory_snapshot=memory_snapshot,
            skills_index=skills_index,
        )
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": current_message})
        return messages

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message (optionally with tool_calls)."""
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)
        return messages

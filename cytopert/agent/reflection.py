"""Reflection hook: a brief LLM turn that updates memory / proposes skills / advances chain status.

This is CytoPert's analogue of Hermes' "agent-curated memory with periodic nudges":
after a non-trivial workflow turn, the agent runs ONE additional LLM call (with a
fresh system prompt — not the main conversation, so the prefix cache stays clean)
and emits a structured JSON describing:

- memory_updates       : list[{action, target, content?, old_text?}]
- skill_proposals      : list[{name, category, description, content}]  (always staged)
- chain_status_updates : list[{chain_id, status, evidence_ids, note}]

Skills are written to ``~/.cytopert/skills/.staged/`` and only become live when the
researcher runs ``cytopert skills accept <name>``. This avoids polluting the procedural
memory with low-quality auto-generated drafts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from cytopert.data.evidence_builder import build_evidence_summary

if TYPE_CHECKING:
    from cytopert.agent.loop import AgentLoop


REFLECTION_SYSTEM_PROMPT = """\
You are CytoPert's reflection module. Your job is to look back on the turn that just
finished and decide what to remember.

Output STRICT JSON (no markdown fences, no commentary) with this schema:
{
  "memory_updates": [
    {"action": "add|replace|remove", "target": "context|researcher|hypothesis_log",
     "content": "<short, information-dense entry>", "old_text": "<unique substring; only for replace/remove>"}
  ],
  "skill_proposals": [
    {"name": "kebab-case-name", "category": "pipelines|reasoning|knowledge",
     "description": "one-line description",
     "content": "---\\nname: ...\\ndescription: ...\\nmetadata:\\n  cytopert:\\n    category: ...\\n---\\n# Title\\n## When to Use\\n## Procedure\\n## Pitfalls\\n## Verification"}
  ],
  "chain_status_updates": [
    {"chain_id": "<existing id>", "status": "proposed|supported|refuted|superseded",
     "evidence_ids": ["..."], "note": "<reason for transition>"}
  ]
}

Rules:
- Be CONSERVATIVE. Empty arrays are fine. Most turns produce no updates.
- Only propose a skill when the workflow used >= 4 tools successfully and the procedure is reusable.
- Only update chain status when explicit new evidence justifies the transition.
- Memory entries must be < 250 chars each and information-dense (no fluff).
- Never invent evidence IDs or chain IDs. Use only the IDs listed below.
- If unsure, return {"memory_updates": [], "skill_proposals": [], "chain_status_updates": []}.
"""


def should_reflect(
    *,
    tool_calls_count: int,
    chains_touched: list[str],
    new_evidence_ids: list[str],
    user_feedback: str | None = None,
    triggers: dict[str, int] | None = None,
) -> bool:
    """Decide whether this turn warrants a reflection LLM call."""
    cfg = triggers or {}
    min_tools = int(cfg.get("min_tool_calls", 5))
    min_evidence = int(cfg.get("min_evidence_entries", 3))
    if tool_calls_count >= min_tools:
        return True
    if len(new_evidence_ids) >= min_evidence:
        return True
    if chains_touched:
        return True
    if user_feedback:
        return True
    return False


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACES_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_reflection_json(text: str | None) -> dict[str, Any]:
    """Tolerant JSON extractor: handle plain JSON, fenced JSON, or pre/post text."""
    if not text:
        return {}
    text = text.strip()
    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)
    braces = _BRACES_RE.search(text)
    if braces:
        candidates.append(braces.group(0))
    for c in candidates:
        try:
            data = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def _build_reflection_user_prompt(
    user_message: str,
    final_response: str,
    tool_calls_count: int,
    chains_touched: list[str],
    new_evidence_ids: list[str],
    memory_snapshot: str,
    evidence_summary: str | None,
    user_feedback: str | None,
) -> str:
    parts = [
        "## Last Turn",
        f"User message:\n{user_message}",
        f"\nAssistant final response (truncated):\n{final_response[:1200]}",
        f"\nTool calls in this turn: {tool_calls_count}",
        f"Chains touched: {chains_touched or 'none'}",
        f"New evidence IDs (cite ONLY these in updates): {new_evidence_ids or 'none'}",
    ]
    if user_feedback:
        parts.append(f"\nResearcher feedback (verbatim):\n{user_feedback}")
    if evidence_summary:
        parts.append("\n## Recent Evidence Summary\n" + evidence_summary)
    parts.append("\n## Current Memory Snapshot\n" + (memory_snapshot or "(empty)"))
    parts.append(
        "\nReturn the JSON described in the system prompt. "
        "Empty arrays are perfectly acceptable."
    )
    return "\n".join(parts)


def apply_reflection(loop: "AgentLoop", payload: dict[str, Any]) -> dict[str, Any]:
    """Apply parsed reflection updates to memory / skills (staged) / chain store.

    Returns a summary dict useful for tests and logging.
    """
    summary = {"memory_applied": 0, "skills_staged": 0, "chains_updated": 0, "errors": []}

    for update in payload.get("memory_updates", []) or []:
        if not isinstance(update, dict):
            continue
        action = update.get("action")
        target = update.get("target")
        try:
            if action == "add" and target:
                res = loop.memory.add(target, update.get("content", ""))
            elif action == "replace" and target:
                res = loop.memory.replace(target, update.get("old_text", ""), update.get("content", ""))
            elif action == "remove" and target:
                res = loop.memory.remove(target, update.get("old_text", ""))
            else:
                continue
            if res.success:
                summary["memory_applied"] += 1
            else:
                summary["errors"].append(f"memory.{action}({target}): {res.message}")
        except (ValueError, KeyError) as e:
            summary["errors"].append(f"memory.{action}({target}): {e}")

    for proposal in payload.get("skill_proposals", []) or []:
        if not isinstance(proposal, dict):
            continue
        name = proposal.get("name")
        content = proposal.get("content")
        category = proposal.get("category", "uncategorized")
        if not name or not content:
            continue
        try:
            loop.skills.create(name, content, category=category, staged=True)
            summary["skills_staged"] += 1
        except (FileExistsError, ValueError) as e:
            summary["errors"].append(f"skill_propose({name}): {e}")

    for upd in payload.get("chain_status_updates", []) or []:
        if not isinstance(upd, dict):
            continue
        chain_id = upd.get("chain_id")
        status = upd.get("status")
        if not chain_id or not status:
            continue
        try:
            loop.chains.update_status(
                chain_id=chain_id,
                status=status,
                evidence_ids=list(upd.get("evidence_ids") or []),
                note=upd.get("note", ""),
            )
            summary["chains_updated"] += 1
        except (KeyError, ValueError) as e:
            summary["errors"].append(f"chain_status({chain_id}): {e}")

    return summary


async def maybe_reflect(
    *,
    loop: "AgentLoop",
    session_key: str,
    user_message: str,
    final_response: str,
    tool_calls_count: int,
    chains_touched: list[str],
    new_evidence_ids: list[str],
    triggers: dict[str, int] | None = None,
    user_feedback: str | None = None,
) -> dict[str, Any] | None:
    """Run the reflection turn if triggered. Returns the apply summary, or None."""
    if not should_reflect(
        tool_calls_count=tool_calls_count,
        chains_touched=chains_touched,
        new_evidence_ids=new_evidence_ids,
        user_feedback=user_feedback,
        triggers=triggers,
    ):
        return None

    memory_snapshot = loop.memory.render_snapshot()
    evidence_summary = (
        build_evidence_summary(loop._evidence_store) if loop._evidence_store else None
    )
    user_prompt = _build_reflection_user_prompt(
        user_message=user_message,
        final_response=final_response,
        tool_calls_count=tool_calls_count,
        chains_touched=chains_touched,
        new_evidence_ids=new_evidence_ids,
        memory_snapshot=memory_snapshot,
        evidence_summary=evidence_summary,
        user_feedback=user_feedback,
    )
    response = await loop.provider.chat(
        messages=[
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=None,
        model=loop.model,
        max_tokens=1500,
        temperature=0.0,
    )
    if response.finish_reason == "error":
        # Surface the failure so a long-running deployment does not look
        # like reflection just never fired. Truncate the body so a
        # multi-kilobyte error message does not flood the logs.
        body = (response.content or "")[:200]
        logging.getLogger(__name__).warning(
            "reflection LLM call returned finish_reason=error: %s", body
        )
        return None
    payload = parse_reflection_json(response.content)
    if not payload:
        body = (response.content or "")[:200]
        logging.getLogger(__name__).warning(
            "reflection LLM response could not be parsed as JSON: %s", body
        )
        return None
    return apply_reflection(loop, payload)

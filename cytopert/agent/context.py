"""Context builder for assembling agent prompts."""

from pathlib import Path
from typing import Any


class ContextBuilder:
    """Builds the context (system prompt + messages) for the CytoPert agent."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def build_system_prompt(self, evidence_summary: str | None = None) -> str:
        """Build the system prompt including evidence and constraint instructions."""
        parts = [self._get_identity()]
        if evidence_summary:
            parts.append(f"# Evidence Summary\n\n{evidence_summary}")
        parts.append(self._get_constraint_instructions())
        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        return """# CytoPert 🧬

You are CytoPert, an AI assistant for cell perturbation differential response mechanism parsing.
You help researchers identify trigger state conditions and decisive regulatory nodes, and output mechanism chains that can be supported or refuted by experiments.

## Guidelines
- Generate an execution plan (which tools to call, data/conditions) before running heavy computations; wait for researcher confirmation when appropriate.
- Every mechanism conclusion must cite traceable evidence (data or authoritative knowledge).
- Reasoning must stay within given pathway hierarchy and regulatory network topology.
- Be concise and cite evidence IDs in your conclusions.

## Tool Use
- If data is needed, call `census_query` (for cellxgene Census) or `load_local_h5ad` (for local h5ad).
- If analysis is needed, use scanpy/pertpy/decoupler tools rather than describing steps abstractly.
- After tool calls, summarize results and state next steps or ask for confirmation.
- If the user explicitly requests a specific tool call, you MUST call that tool with the given parameters.

## Evidence Integrity
- Do NOT invent citations, datasets, or evidence IDs.
- Only cite evidence IDs that actually exist in the evidence store or tool outputs.
- If you lack evidence, say so and ask for the missing data or permission to run tools.
"""

    def _get_constraint_instructions(self) -> str:
        return """# Constraints
- Only reason within the given pathway hierarchy and regulatory network topology.
- Each key conclusion must reference an evidence entry ID (data or knowledge).
- Output mechanism chains with verification readouts and priority (e.g. P1/P2) when requested.
"""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        evidence_summary: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        messages = []
        system_prompt = self.build_system_prompt(evidence_summary)
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

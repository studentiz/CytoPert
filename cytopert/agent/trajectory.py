"""Trajectory saving for downstream training / evaluation.

Adapted from NousResearch/hermes-agent
24342813fe2196335ac8e510e8f59f716197d0e8:agent/trajectory.py (MIT License).
See docs/hermes-borrowing.md for the per-module diff rationale.

Differences from upstream:
    * Drop the ``<scratchpad>`` / ``<think>`` rewrite helpers; CytoPert
      does not run thinking-mode models in this code path.
    * Carry CytoPert-specific metadata in the JSONL header
      (``evidence_ids`` and ``chains_touched``) so a future training
      pipeline can compute reward signals from the chain-of-citations.

The default output directory is ``~/.cytopert/trajectories/``. Callers
that want to save trajectories elsewhere can pass ``filename`` to
``save_trajectory``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from cytopert.session.manager import Session
from cytopert.utils.helpers import ensure_dir, get_data_path

logger = logging.getLogger(__name__)

#: Mapping from internal role names to the ShareGPT vocabulary the
#: training stack expects. ``tool`` becomes ``tool`` (kept verbatim) so
#: tool calls remain identifiable in the resulting JSONL.
_ROLE_TO_SHAREGPT = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
    "tool": "tool",
}


def trajectories_dir() -> Path:
    """Return the on-disk directory where trajectories are written."""
    return ensure_dir(get_data_path() / "trajectories")


def convert_session_to_sharegpt(
    session: Session,
    *,
    include_system: bool = False,
) -> list[dict[str, str]]:
    """Convert a Session's message log to the ShareGPT shape.

    The CytoPert AgentLoop only stores user / assistant pairs in
    ``session.messages`` (system / tool messages live in the per-call
    message buffer the AgentLoop builds on the fly), so the returned
    list is mostly ``human`` / ``gpt`` alternations. ``include_system``
    is kept for forward compatibility with future versions that
    persist tool turns into the session log.
    """
    out: list[dict[str, str]] = []
    for msg in session.messages:
        role = msg.get("role", "")
        if role == "system" and not include_system:
            continue
        out.append(
            {
                "from": _ROLE_TO_SHAREGPT.get(role, role),
                "value": str(msg.get("content", "")),
            }
        )
    return out


def save_trajectory(
    trajectory: list[dict[str, str]],
    *,
    model: str,
    completed: bool,
    evidence_ids: list[str] | None = None,
    chains_touched: list[str] | None = None,
    session_key: str | None = None,
    filename: str | Path | None = None,
) -> Path:
    """Append one trajectory entry to a JSONL file and return its path.

    The default filename routes to ``trajectory_samples.jsonl`` (when
    ``completed=True``) or ``failed_trajectories.jsonl``. Each entry
    carries CytoPert-specific metadata so a downstream RL / SFT pipeline
    can reward chain-of-citation correctness without having to re-derive
    it from the raw conversation.
    """
    if filename is None:
        target = trajectories_dir() / (
            "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
        )
    else:
        target = Path(filename).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "conversations": trajectory,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "completed": completed,
        "evidence_ids": list(evidence_ids or []),
        "chains_touched": list(chains_touched or []),
        "session_key": session_key,
    }
    try:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to append trajectory to %s: %s", target, exc)
    return target

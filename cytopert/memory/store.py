"""MemoryStore: three markdown files acting as CytoPert's curated semantic memory.

Inspired by Hermes Agent's MEMORY.md / USER.md, with one CytoPert-specific addition:

- ``context``      : agent-curated environment / tool habits / conventions      (~2200 chars)
- ``researcher``   : researcher profile / preferences / output format            (~1375 chars)
- ``hypothesis_log``: compact log of active mechanism hypotheses across sessions (~3000 chars)

Each store is plain markdown with entries separated by ``§`` (section sign). The
agent manipulates entries via ``add`` / ``replace`` / ``remove`` (substring matching).
A frozen snapshot is rendered into the system prompt at session start; in-session
edits are persisted immediately but never rewritten into the live system prompt
(this preserves the LLM prefix cache, mirroring Hermes' design).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENTRY_DELIMITER = "§"

MEMORY_TARGETS = ("context", "researcher", "hypothesis_log")

DEFAULT_LIMITS: dict[str, int] = {
    "context": 2200,
    "researcher": 1375,
    "hypothesis_log": 3000,
}

DEFAULT_HEADERS: dict[str, str] = {
    "context": "CONTEXT (CytoPert agent's environment / tool habits / conventions)",
    "researcher": "RESEARCHER (preferences, focus organisms / tissues, output format)",
    "hypothesis_log": "HYPOTHESIS LOG (active mechanism chains and their status)",
}

DEFAULT_FILENAMES: dict[str, str] = {
    "context": "CONTEXT.md",
    "researcher": "RESEARCHER.md",
    "hypothesis_log": "HYPOTHESIS_LOG.md",
}


@dataclass
class MemoryResult:
    """Outcome of a memory mutation."""

    success: bool
    target: str
    message: str = ""
    entries: list[str] = field(default_factory=list)
    usage_chars: int = 0
    usage_limit: int = 0

    @property
    def usage_pct(self) -> int:
        if not self.usage_limit:
            return 0
        return int(round(100.0 * self.usage_chars / self.usage_limit))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "target": self.target,
            "message": self.message,
            "usage": f"{self.usage_chars}/{self.usage_limit}",
            "usage_pct": self.usage_pct,
            "entries": self.entries,
        }


class MemoryStore:
    """File-backed memory with three targets and per-target character limits."""

    def __init__(
        self,
        memory_dir: Path,
        limits: dict[str, int] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.headers = {**DEFAULT_HEADERS, **(headers or {})}

    def _validate_target(self, target: str) -> None:
        if target not in MEMORY_TARGETS:
            raise ValueError(
                f"Unknown memory target {target!r}; expected one of {MEMORY_TARGETS}"
            )

    def _path(self, target: str) -> Path:
        return self.memory_dir / DEFAULT_FILENAMES[target]

    def read(self, target: str) -> str:
        self._validate_target(target)
        path = self._path(target)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def entries(self, target: str) -> list[str]:
        raw = self.read(target).strip()
        if not raw:
            return []
        return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]

    def usage(self, target: str) -> tuple[int, int]:
        return len(self.read(target)), self.limits[target]

    def _write_entries(self, target: str, entries: list[str]) -> None:
        body = f"\n{ENTRY_DELIMITER}\n".join(e.strip() for e in entries if e.strip())
        self._path(target).write_text(body, encoding="utf-8")

    def add(self, target: str, content: str, *, dedupe: bool = True) -> MemoryResult:
        self._validate_target(target)
        content = (content or "").strip()
        if not content:
            return self._fail(target, "Empty content")
        existing = self.entries(target)
        if dedupe and content in existing:
            usage, limit = self.usage(target)
            return MemoryResult(
                success=True,
                target=target,
                message="Duplicate entry; nothing added.",
                entries=existing,
                usage_chars=usage,
                usage_limit=limit,
            )
        new_entries = existing + [content]
        candidate = self._render_entries(new_entries)
        if len(candidate) > self.limits[target]:
            return self._fail(
                target,
                (
                    f"Memory at {len(self._render_entries(existing))}/{self.limits[target]} chars. "
                    f"Adding this entry ({len(content)} chars) would exceed the limit. "
                    "Replace or remove existing entries first."
                ),
                entries=existing,
            )
        self._write_entries(target, new_entries)
        usage = len(candidate)
        return MemoryResult(
            success=True,
            target=target,
            message="Added.",
            entries=new_entries,
            usage_chars=usage,
            usage_limit=self.limits[target],
        )

    def replace(self, target: str, old_text: str, content: str) -> MemoryResult:
        self._validate_target(target)
        old_text = (old_text or "").strip()
        content = (content or "").strip()
        if not old_text or not content:
            return self._fail(target, "Both old_text and content are required")
        existing = self.entries(target)
        matches = [i for i, e in enumerate(existing) if old_text in e]
        if not matches:
            return self._fail(target, f"No entry matched substring {old_text!r}", entries=existing)
        if len(matches) > 1:
            return self._fail(
                target,
                f"Substring {old_text!r} matched {len(matches)} entries; provide a more specific old_text.",
                entries=existing,
            )
        new_entries = list(existing)
        new_entries[matches[0]] = content
        candidate = self._render_entries(new_entries)
        if len(candidate) > self.limits[target]:
            return self._fail(
                target,
                "Replacement would exceed character limit; shorten content or consolidate first.",
                entries=existing,
            )
        self._write_entries(target, new_entries)
        return MemoryResult(
            success=True,
            target=target,
            message="Replaced.",
            entries=new_entries,
            usage_chars=len(candidate),
            usage_limit=self.limits[target],
        )

    def remove(self, target: str, old_text: str) -> MemoryResult:
        self._validate_target(target)
        old_text = (old_text or "").strip()
        if not old_text:
            return self._fail(target, "old_text required")
        existing = self.entries(target)
        matches = [i for i, e in enumerate(existing) if old_text in e]
        if not matches:
            return self._fail(target, f"No entry matched substring {old_text!r}", entries=existing)
        if len(matches) > 1:
            return self._fail(
                target,
                f"Substring {old_text!r} matched {len(matches)} entries; provide a more specific old_text.",
                entries=existing,
            )
        new_entries = [e for i, e in enumerate(existing) if i != matches[0]]
        self._write_entries(target, new_entries)
        candidate = self._render_entries(new_entries)
        return MemoryResult(
            success=True,
            target=target,
            message="Removed.",
            entries=new_entries,
            usage_chars=len(candidate),
            usage_limit=self.limits[target],
        )

    def clear(self, target: str | None = None) -> None:
        targets = [target] if target else list(MEMORY_TARGETS)
        for t in targets:
            self._validate_target(t)
            p = self._path(t)
            if p.exists():
                p.unlink()

    def render_snapshot(self) -> str:
        """Frozen, system-prompt-friendly rendering of all three stores."""
        blocks: list[str] = []
        for target in MEMORY_TARGETS:
            entries = self.entries(target)
            chars, limit = self.usage(target)
            pct = int(round(100.0 * chars / limit)) if limit else 0
            header = self.headers[target]
            sep = "=" * 78
            block_lines = [sep, f"{header}  [{pct}% — {chars}/{limit} chars]", sep]
            if entries:
                block_lines.append(f"\n{ENTRY_DELIMITER}\n".join(entries))
            else:
                block_lines.append("(empty)")
            blocks.append("\n".join(block_lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _render_entries(entries: list[str]) -> str:
        return f"\n{ENTRY_DELIMITER}\n".join(e.strip() for e in entries if e.strip())

    def _fail(self, target: str, message: str, entries: list[str] | None = None) -> MemoryResult:
        chars, limit = self.usage(target)
        return MemoryResult(
            success=False,
            target=target,
            message=message,
            entries=entries or self.entries(target),
            usage_chars=chars,
            usage_limit=limit,
        )


_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2028-\u202f\u2066-\u2069]")


def sanitize_entry(text: str) -> str:
    """Strip invisible Unicode characters often used in prompt-injection attempts."""
    return _INVISIBLE_RE.sub("", text or "").strip()

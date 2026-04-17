"""Session management for conversation history.

Each session is a JSONL file: one metadata header line (``_type=metadata``)
plus one JSON-encoded message per subsequent line. The header now round-trips
``created_at`` / ``updated_at`` / ``metadata`` so tools that inspect a session
on disk get the timestamps the agent originally wrote -- the legacy ``_load``
silently dropped them.

The default sessions directory is still ``~/.cytopert/sessions`` so existing
on-disk sessions remain readable. Callers can pass a ``workspace_label`` to
isolate sessions per project (e.g. one ``CYTOPERT_HOME`` per study).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cytopert.utils.helpers import ensure_dir, get_data_path, safe_filename

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A conversation session with message history."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """Get message history for LLM context (role, content only)."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": m["role"], "content": m["content"]} for m in recent]


class SessionManager:
    """Manages conversation sessions stored as JSONL files."""

    def __init__(self, workspace: Path, *, workspace_label: str | None = None) -> None:
        self.workspace = workspace
        base = get_data_path() / "sessions"
        if workspace_label:
            base = base / safe_filename(workspace_label.replace(":", "_"))
        self.sessions_dir = ensure_dir(base)
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """Get an existing session or create a new one."""
        if key in self._cache:
            return self._cache[key]
        session = self._load(key)
        if session is None:
            session = Session(key=key)
        self._cache[key] = session
        return session

    @staticmethod
    def _parse_iso(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _load(self, key: str) -> Session | None:
        path = self._get_session_path(key)
        if not path.exists():
            return None
        try:
            messages: list[dict[str, Any]] = []
            created_at: datetime | None = None
            updated_at: datetime | None = None
            metadata: dict[str, Any] = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        # First-line metadata: round-trip into the Session
                        # so re-loaded sessions retain their original
                        # created_at and any custom metadata the agent
                        # attached (e.g. plan_mode in stage 4).
                        created_at = self._parse_iso(data.get("created_at")) or created_at
                        updated_at = self._parse_iso(data.get("updated_at")) or updated_at
                        meta = data.get("metadata")
                        if isinstance(meta, dict):
                            metadata = meta
                        continue
                    messages.append(data)
        except (OSError, json.JSONDecodeError) as exc:
            # The legacy implementation returned None and the agent silently
            # treated the bad session as new. Log and start fresh, so the
            # corruption is at least visible in the logs.
            logger.warning("Session %s JSONL is unreadable (%s); starting fresh", key, exc)
            return None
        kwargs: dict[str, Any] = {"key": key, "messages": messages, "metadata": metadata}
        if created_at is not None:
            kwargs["created_at"] = created_at
        if updated_at is not None:
            kwargs["updated_at"] = updated_at
        return Session(**kwargs)

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)
        with open(path, "w", encoding="utf-8") as f:
            meta = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._cache[session.key] = session

    def reset(self, key: str) -> None:
        """Clear a session from disk and cache."""
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
        if key in self._cache:
            del self._cache[key]

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return basic metadata for every session on disk in this directory."""
        out: list[dict[str, Any]] = []
        for path in sorted(self.sessions_dir.glob("*.jsonl")):
            key = path.stem
            try:
                with open(path, encoding="utf-8") as f:
                    first = f.readline().strip()
                meta = json.loads(first) if first else {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read metadata from %s: %s", path, exc)
                meta = {}
            if not isinstance(meta, dict) or meta.get("_type") != "metadata":
                meta = {}
            out.append(
                {
                    "key": key,
                    "path": str(path),
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                    "metadata": meta.get("metadata", {}),
                }
            )
        return out

    def export_session(self, key: str, target: Path | str) -> Path:
        """Copy the JSONL file for ``key`` to ``target`` and return the new path."""
        src = self._get_session_path(key)
        if not src.exists():
            raise FileNotFoundError(f"No session named {key!r} at {src}")
        dst = Path(target).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return dst

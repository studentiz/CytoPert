"""EvidenceDB: SQLite + FTS5 store for cross-session evidence retrieval.

This is CytoPert's "episodic memory" — every EvidenceEntry produced by a tool call
is persisted with its session id and timestamp, then made searchable across all
future sessions via FTS5 (summary / genes / pathways / tool_name / source).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from cytopert.data.models import EvidenceEntry, EvidenceType
from cytopert.persistence.schema import ALL_DDL

logger = logging.getLogger(__name__)


def _serialize_list(value: list[str] | None) -> str:
    return json.dumps(value or [], ensure_ascii=False)


def _deserialize_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        out = json.loads(value)
        if isinstance(out, list):
            return [str(x) for x in out]
    except (json.JSONDecodeError, TypeError) as exc:
        # A bad JSON column would otherwise silently produce an empty list,
        # making evidence look gene-less to the model. Surface the corruption.
        logger.warning("Evidence column is not valid JSON: %s; raw=%r", exc, value[:80])
    return []


def _row_to_entry(row: sqlite3.Row) -> EvidenceEntry:
    return EvidenceEntry(
        id=row["id"],
        type=EvidenceType(row["type"]) if row["type"] else EvidenceType.DATA,
        source=row["source"] or "",
        supports=bool(row["supports"]) if row["supports"] is not None else True,
        confidence=float(row["confidence"]) if row["confidence"] is not None else 0.5,
        genes=_deserialize_list(row["genes_json"]),
        pathways=_deserialize_list(row["pathways_json"]),
        state_conditions=_deserialize_list(row["state_conditions_json"]),
        summary=row["summary"] or "",
        tool_name=row["tool_name"],
        extra=json.loads(row["extra_json"]) if row["extra_json"] else {},
    )


class EvidenceDB:
    """SQLite-backed evidence store with FTS5 search."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            for ddl in ALL_DDL:
                conn.executescript(ddl)
            conn.commit()

    def add(self, entry: EvidenceEntry, session_id: str = "") -> None:
        """Insert or replace an evidence entry. FTS5 mirror is updated by triggers."""
        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evidence_entries
                    (id, session_id, type, source, supports, confidence, summary,
                     genes_json, pathways_json, state_conditions_json, tool_name,
                     extra_json, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id=excluded.session_id,
                    type=excluded.type,
                    source=excluded.source,
                    supports=excluded.supports,
                    confidence=excluded.confidence,
                    summary=excluded.summary,
                    genes_json=excluded.genes_json,
                    pathways_json=excluded.pathways_json,
                    state_conditions_json=excluded.state_conditions_json,
                    tool_name=excluded.tool_name,
                    extra_json=excluded.extra_json
                """,
                (
                    entry.id,
                    session_id,
                    entry.type.value if isinstance(entry.type, EvidenceType) else str(entry.type),
                    entry.source,
                    1 if entry.supports else 0,
                    float(entry.confidence),
                    entry.summary,
                    _serialize_list(entry.genes),
                    _serialize_list(entry.pathways),
                    _serialize_list(entry.state_conditions),
                    entry.tool_name,
                    json.dumps(entry.extra or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()

    def add_many(self, entries: list[EvidenceEntry], session_id: str = "") -> int:
        for e in entries:
            self.add(e, session_id=session_id)
        return len(entries)

    def get(self, evidence_id: str) -> EvidenceEntry | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evidence_entries WHERE id = ?", (evidence_id,)
            ).fetchone()
        return _row_to_entry(row) if row else None

    def recent(self, limit: int = 50, session_id: str | None = None) -> list[EvidenceEntry]:
        sql = "SELECT * FROM evidence_entries"
        params: list[Any] = []
        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_entry(r) for r in rows]

    def search(
        self,
        query: str | None = None,
        gene: str | None = None,
        pathway: str | None = None,
        tissue: str | None = None,
        tool_name: str | None = None,
        top_k: int = 20,
    ) -> list[EvidenceEntry]:
        """Full-text + structured filter search.

        - `query` is matched against summary/genes/pathways/source/tool_name via FTS5.
        - `gene` / `pathway` / `tissue` apply substring filters on JSON columns / source.
        - `tool_name` is an exact equality filter.
        """
        with self._lock, self._connect() as conn:
            ids: list[str] | None = None
            if query:
                fts_query = self._build_fts_query(query)
                rows = conn.execute(
                    "SELECT id FROM evidence_fts WHERE evidence_fts MATCH ? LIMIT ?",
                    (fts_query, int(top_k) * 4),
                ).fetchall()
                ids = [r["id"] for r in rows]
                if not ids:
                    return []

            sql = "SELECT * FROM evidence_entries WHERE 1=1"
            params: list[Any] = []
            if ids is not None:
                placeholders = ",".join("?" for _ in ids)
                sql += f" AND id IN ({placeholders})"
                params.extend(ids)
            if gene:
                sql += " AND genes_json LIKE ?"
                params.append(f"%{gene}%")
            if pathway:
                sql += " AND pathways_json LIKE ?"
                params.append(f"%{pathway}%")
            if tissue:
                sql += " AND (state_conditions_json LIKE ? OR source LIKE ? OR summary LIKE ?)"
                params.extend([f"%{tissue}%", f"%{tissue}%", f"%{tissue}%"])
            if tool_name:
                sql += " AND tool_name = ?"
                params.append(tool_name)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(int(top_k))

            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_entry(r) for r in rows]

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Sanitize a free-text query for FTS5 MATCH — quote each token to avoid syntax errors."""
        tokens = [t for t in query.replace('"', " ").split() if t]
        if not tokens:
            return '""'
        return " ".join(f'"{t}"' for t in tokens)

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM evidence_entries").fetchone()
        return int(row["n"]) if row else 0

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM evidence_entries")
            conn.commit()

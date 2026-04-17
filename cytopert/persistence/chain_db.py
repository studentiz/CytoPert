"""ChainStore: persistent MechanismChain lifecycle store.

Each MechanismChain has a status state machine:
    proposed -> supported | refuted | superseded

Events for every status transition are appended to:
    - SQLite `chain_events` table (for queryable history)
    - JSONL audit file at <chains_dir>/chain_<id>.jsonl (for human-readable trail)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from cytopert.data.models import MechanismChain, MechanismLink
from cytopert.persistence.schema import ALL_DDL, CHAIN_STATUSES

logger = logging.getLogger(__name__)


def _serialize_links(links: list[MechanismLink]) -> str:
    return json.dumps([link.model_dump() for link in links], ensure_ascii=False)


def _deserialize_links(value: str | None, *, chain_id: str = "") -> list[MechanismLink]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        # Bad JSON in the column would otherwise silently produce an empty
        # chain -- log so the corruption surfaces in a real run.
        logger.warning("Chain %s links_json is not valid JSON: %s", chain_id or "?", exc)
        return []
    out: list[MechanismLink] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            out.append(MechanismLink(**item))
    return out


def _row_to_chain(row: sqlite3.Row) -> MechanismChain:
    return MechanismChain(
        id=row["id"],
        summary=row["summary"] or "",
        priority=row["priority"] or "P2",
        status=row["status"] or "proposed",
        verification_readout=row["verification_readout"] or "",
        evidence_ids=json.loads(row["evidence_ids_json"]) if row["evidence_ids_json"] else [],
        links=_deserialize_links(row["links_json"], chain_id=row["id"]),
    )


class ChainStore:
    """Persistent MechanismChain registry with lifecycle events."""

    def __init__(self, db_path: Path, chains_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chains_dir = Path(chains_dir)
        self.chains_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            for ddl in ALL_DDL:
                conn.executescript(ddl)
            conn.commit()

    def upsert(self, chain: MechanismChain, status: str = "proposed", note: str = "") -> str:
        """Create or update a chain. Always records an event in the JSONL trail."""
        if not chain.id:
            chain.id = self._next_chain_id()
        if status not in CHAIN_STATUSES:
            raise ValueError(f"Invalid status {status!r}; expected one of {sorted(CHAIN_STATUSES)}")
        now = datetime.utcnow().isoformat()
        existing = self.get(chain.id)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chains
                    (id, summary, status, priority, verification_readout,
                     evidence_ids_json, links_json, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    summary=excluded.summary,
                    status=excluded.status,
                    priority=excluded.priority,
                    verification_readout=excluded.verification_readout,
                    evidence_ids_json=excluded.evidence_ids_json,
                    links_json=excluded.links_json,
                    updated_at=excluded.updated_at
                """,
                (
                    chain.id,
                    chain.summary,
                    status,
                    chain.priority,
                    chain.verification_readout,
                    json.dumps(chain.evidence_ids, ensure_ascii=False),
                    _serialize_links(chain.links),
                    now,
                    now,
                ),
            )
            conn.commit()
        event = {
            "event_type": "create" if existing is None else "update",
            "status": status,
            "evidence_ids": list(chain.evidence_ids),
            "note": note,
            "summary": chain.summary,
            "priority": chain.priority,
        }
        self._append_event(chain.id, event)
        return chain.id

    def update_status(
        self,
        chain_id: str,
        status: str,
        evidence_ids: list[str] | None = None,
        note: str = "",
    ) -> MechanismChain:
        """Transition chain status. Appends event with the supplied evidence_ids."""
        if status not in CHAIN_STATUSES:
            raise ValueError(f"Invalid status {status!r}; expected one of {sorted(CHAIN_STATUSES)}")
        chain = self.get(chain_id)
        if chain is None:
            raise KeyError(f"Chain {chain_id!r} not found")
        evidence_ids = list(evidence_ids or [])
        merged_evidence = list(dict.fromkeys([*chain.evidence_ids, *evidence_ids]))
        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE chains
                SET status = ?, evidence_ids_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, json.dumps(merged_evidence, ensure_ascii=False), now, chain_id),
            )
            conn.commit()
        event = {
            "event_type": "status_change",
            "status": status,
            "evidence_ids": evidence_ids,
            "note": note,
        }
        self._append_event(chain_id, event)
        chain.evidence_ids = merged_evidence
        return chain

    def get(self, chain_id: str) -> MechanismChain | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM chains WHERE id = ?", (chain_id,)).fetchone()
        return _row_to_chain(row) if row else None

    def get_status(self, chain_id: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT status FROM chains WHERE id = ?", (chain_id,)).fetchone()
        return row["status"] if row else None

    def list(
        self,
        status: str | None = None,
        gene: str | None = None,
        limit: int = 100,
    ) -> list[tuple[MechanismChain, str]]:
        """Return (chain, status) pairs, most recently updated first."""
        sql = "SELECT * FROM chains WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if gene:
            sql += " AND (links_json LIKE ? OR summary LIKE ?)"
            params.extend([f"%{gene}%", f"%{gene}%"])
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [(_row_to_chain(r), r["status"]) for r in rows]

    def events(self, chain_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, status, evidence_ids_json, note, payload_json, created_at
                FROM chain_events
                WHERE chain_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (chain_id, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "event_type": r["event_type"],
                    "status": r["status"],
                    "evidence_ids": json.loads(r["evidence_ids_json"]) if r["evidence_ids_json"] else [],
                    "note": r["note"] or "",
                    "payload": json.loads(r["payload_json"]) if r["payload_json"] else {},
                    "created_at": r["created_at"],
                }
            )
        return out

    def _append_event(self, chain_id: str, event: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        payload = {k: v for k, v in event.items() if k not in {"event_type", "status", "evidence_ids", "note"}}
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chain_events
                    (chain_id, event_type, status, evidence_ids_json, note, payload_json, created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    chain_id,
                    event.get("event_type", "update"),
                    event.get("status"),
                    json.dumps(event.get("evidence_ids", []), ensure_ascii=False),
                    event.get("note", ""),
                    json.dumps(payload, ensure_ascii=False) if payload else None,
                    now,
                ),
            )
            conn.commit()
        path = self.chains_dir / f"chain_{chain_id}.jsonl"
        record = {"timestamp": now, **event}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _next_chain_id(self) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chains").fetchone()
        n = int(row["n"]) if row else 0
        return f"chain_{n + 1:04d}"

    def count(self, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM chains"
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        with self._lock, self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0

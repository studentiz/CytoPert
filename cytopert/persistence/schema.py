"""SQLite schema for CytoPert persistence (evidence + chains)."""

EVIDENCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evidence_entries (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    type TEXT,
    source TEXT,
    supports INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    summary TEXT,
    genes_json TEXT,
    pathways_json TEXT,
    state_conditions_json TEXT,
    tool_name TEXT,
    extra_json TEXT,
    created_at TEXT
);
"""

EVIDENCE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    id UNINDEXED,
    summary,
    genes,
    pathways,
    source,
    tool_name,
    tokenize='unicode61'
);
"""

EVIDENCE_FTS_TRIGGERS_SQL = [
    """
    CREATE TRIGGER IF NOT EXISTS evidence_fts_ai AFTER INSERT ON evidence_entries BEGIN
        INSERT INTO evidence_fts(id, summary, genes, pathways, source, tool_name)
        VALUES (
            new.id,
            COALESCE(new.summary, ''),
            COALESCE(new.genes_json, ''),
            COALESCE(new.pathways_json, ''),
            COALESCE(new.source, ''),
            COALESCE(new.tool_name, '')
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS evidence_fts_ad AFTER DELETE ON evidence_entries BEGIN
        DELETE FROM evidence_fts WHERE id = old.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS evidence_fts_au AFTER UPDATE ON evidence_entries BEGIN
        DELETE FROM evidence_fts WHERE id = old.id;
        INSERT INTO evidence_fts(id, summary, genes, pathways, source, tool_name)
        VALUES (
            new.id,
            COALESCE(new.summary, ''),
            COALESCE(new.genes_json, ''),
            COALESCE(new.pathways_json, ''),
            COALESCE(new.source, ''),
            COALESCE(new.tool_name, '')
        );
    END;
    """,
]

CHAINS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chains (
    id TEXT PRIMARY KEY,
    summary TEXT,
    status TEXT DEFAULT 'proposed',
    priority TEXT DEFAULT 'P2',
    verification_readout TEXT,
    evidence_ids_json TEXT,
    links_json TEXT,
    created_at TEXT,
    updated_at TEXT
);
"""

CHAIN_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS chain_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id TEXT,
    event_type TEXT,
    status TEXT,
    evidence_ids_json TEXT,
    note TEXT,
    payload_json TEXT,
    created_at TEXT
);
"""

CHAIN_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_chain_events_chain ON chain_events(chain_id);",
    "CREATE INDEX IF NOT EXISTS idx_chains_status ON chains(status);",
]

ALL_DDL: list[str] = [
    EVIDENCE_TABLE_SQL,
    EVIDENCE_FTS_SQL,
    *EVIDENCE_FTS_TRIGGERS_SQL,
    CHAINS_TABLE_SQL,
    CHAIN_EVENTS_SQL,
    *CHAIN_INDEXES_SQL,
]


CHAIN_STATUSES = {"proposed", "supported", "refuted", "superseded"}

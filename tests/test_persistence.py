"""Tests for cytopert.persistence (EvidenceDB + ChainStore)."""

from pathlib import Path

import pytest

from cytopert.data.models import EvidenceEntry, EvidenceType, MechanismChain, MechanismLink
from cytopert.persistence.chain_db import ChainStore
from cytopert.persistence.evidence_db import EvidenceDB


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def evidence_db(tmp_db: Path) -> EvidenceDB:
    return EvidenceDB(tmp_db)


def _make_entry(eid: str, summary: str, genes: list[str] | None = None,
                pathways: list[str] | None = None, tool: str = "scanpy_de") -> EvidenceEntry:
    return EvidenceEntry(
        id=eid,
        type=EvidenceType.DATA,
        source="local",
        summary=summary,
        genes=genes or [],
        pathways=pathways or [],
        tool_name=tool,
    )


def test_evidence_add_and_get(evidence_db: EvidenceDB) -> None:
    e = _make_entry("e1", "DE basal vs luminal", ["NFATC1", "ESR1"], ["NOTCH"])
    evidence_db.add(e, session_id="s1")
    out = evidence_db.get("e1")
    assert out is not None
    assert out.summary == "DE basal vs luminal"
    assert out.genes == ["NFATC1", "ESR1"]
    assert out.pathways == ["NOTCH"]
    assert evidence_db.count() == 1


def test_evidence_search_fts(evidence_db: EvidenceDB) -> None:
    evidence_db.add(_make_entry("e1", "DE basal vs luminal NFATC1", ["NFATC1"]))
    evidence_db.add(_make_entry("e2", "Pertpy distance Tnf vs control", ["TNF"]))
    evidence_db.add(_make_entry("e3", "Decoupler enrichment Notch pathway", pathways=["NOTCH", "WNT"]))
    hits = evidence_db.search(query="NFATC1")
    assert any(h.id == "e1" for h in hits)
    hits = evidence_db.search(query="Notch")
    assert any(h.id == "e3" for h in hits)


def test_evidence_search_filters(evidence_db: EvidenceDB) -> None:
    evidence_db.add(_make_entry("e1", "DE basal", ["NFATC1"], ["WNT"], tool="scanpy_de"))
    evidence_db.add(_make_entry("e2", "Pertpy run", ["TNF"], [], tool="pertpy"))
    out_gene = evidence_db.search(gene="NFATC1")
    assert {h.id for h in out_gene} == {"e1"}
    out_pw = evidence_db.search(pathway="WNT")
    assert {h.id for h in out_pw} == {"e1"}
    out_tool = evidence_db.search(tool_name="pertpy")
    assert {h.id for h in out_tool} == {"e2"}


def test_evidence_search_empty_query_returns_no_match(evidence_db: EvidenceDB) -> None:
    evidence_db.add(_make_entry("e1", "anything"))
    assert evidence_db.search(query="zzzzznone") == []


def test_evidence_recent_orders_desc(evidence_db: EvidenceDB) -> None:
    evidence_db.add(_make_entry("e1", "first"), session_id="s1")
    evidence_db.add(_make_entry("e2", "second"), session_id="s1")
    evidence_db.add(_make_entry("e3", "third"), session_id="s2")
    recent = evidence_db.recent(limit=2)
    assert len(recent) == 2
    assert recent[0].id == "e3"
    s1_only = evidence_db.recent(limit=10, session_id="s1")
    assert {x.id for x in s1_only} == {"e1", "e2"}


def test_evidence_upsert_idempotent(evidence_db: EvidenceDB) -> None:
    evidence_db.add(_make_entry("e1", "v1"))
    evidence_db.add(_make_entry("e1", "v2", genes=["NFATC1"]))
    out = evidence_db.get("e1")
    assert out and out.summary == "v2" and out.genes == ["NFATC1"]
    assert evidence_db.count() == 1


def test_chain_store_lifecycle(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "state.db", tmp_path / "chains")
    chain = MechanismChain(
        id="chain_test",
        summary="NFATC1 -> NOTCH -> luminal differentiation",
        evidence_ids=["e1"],
        links=[MechanismLink(from_node="NFATC1", to_node="NOTCH", relation="regulates",
                             evidence_ids=["e1"])],
    )
    cid = store.upsert(chain, status="proposed", note="initial")
    assert cid == "chain_test"
    assert store.get_status(cid) == "proposed"

    store.update_status(cid, "supported", evidence_ids=["e2"], note="exp confirmed")
    assert store.get_status(cid) == "supported"

    chain_back = store.get(cid)
    assert chain_back is not None
    assert set(chain_back.evidence_ids) == {"e1", "e2"}

    events = store.events(cid)
    assert [e["event_type"] for e in events] == ["create", "status_change"]
    assert events[1]["status"] == "supported"

    audit = (tmp_path / "chains" / f"chain_{cid}.jsonl").read_text(encoding="utf-8")
    assert "status_change" in audit and "supported" in audit


def test_chain_store_list_and_filters(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "state.db", tmp_path / "chains")
    store.upsert(MechanismChain(id="c1", summary="A", evidence_ids=["e1"]), status="proposed")
    store.upsert(MechanismChain(id="c2", summary="B about NFATC1", evidence_ids=["e1"]),
                 status="supported")
    store.upsert(MechanismChain(id="c3", summary="C", evidence_ids=["e1"]), status="refuted")

    proposed = store.list(status="proposed")
    assert {c.id for c, _ in proposed} == {"c1"}
    nfatc1 = store.list(gene="NFATC1")
    assert {c.id for c, _ in nfatc1} == {"c2"}
    assert store.count(status="refuted") == 1


def test_chain_store_invalid_status(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "state.db", tmp_path / "chains")
    chain = MechanismChain(id="c1", summary="x", evidence_ids=["e1"])
    store.upsert(chain)
    with pytest.raises(ValueError):
        store.update_status("c1", "bogus", evidence_ids=[])


def test_chain_store_auto_id(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "state.db", tmp_path / "chains")
    chain = MechanismChain(id="", summary="auto", evidence_ids=["e1"])
    cid = store.upsert(chain, status="proposed")
    assert cid.startswith("chain_")
    assert store.get(cid) is not None

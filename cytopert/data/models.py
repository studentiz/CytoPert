"""Pydantic models for evidence entries and mechanism chains."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Type of evidence entry."""

    DATA = "data"
    KNOWLEDGE = "knowledge"


class EvidenceEntry(BaseModel):
    """Structured evidence entry for mechanism reasoning."""

    id: str = Field(..., description="Unique evidence ID for citation")
    type: EvidenceType = Field(..., description="Data evidence or authoritative knowledge")
    source: str = Field("", description="dataset_id, method, or literature reference")
    supports: bool = Field(True, description="Whether this evidence supports the claim")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence score")
    genes: list[str] = Field(default_factory=list, description="Associated genes")
    pathways: list[str] = Field(default_factory=list, description="Associated pathways")
    state_conditions: list[str] = Field(default_factory=list, description="State/condition labels")
    summary: str = Field("", description="Short human-readable summary")
    tool_name: str | None = Field(None, description="Tool that produced this (e.g. scanpy_de)")
    extra: dict[str, Any] = Field(default_factory=dict)


class MechanismLink(BaseModel):
    """A single link in a mechanism chain (e.g. gene A -> pathway -> outcome)."""

    from_node: str = Field("", description="Upstream node (gene/pathway/state)")
    to_node: str = Field("", description="Downstream node")
    relation: str = Field("", description="Type of relation (e.g. regulates, activates)")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence IDs supporting this link")


class MechanismChain(BaseModel):
    """A mechanism chain candidate with verification readouts."""

    id: str = Field("", description="Chain identifier")
    links: list[MechanismLink] = Field(default_factory=list)
    summary: str = Field("", description="Short summary of the chain")
    verification_readout: str = Field("", description="Suggested experimental readout to test")
    priority: str = Field("P2", description="Priority e.g. P1, P2")
    evidence_ids: list[str] = Field(default_factory=list, description="All evidence IDs cited")

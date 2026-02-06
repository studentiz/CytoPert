"""Basic tests for CytoPert."""

import pytest

from cytopert import __version__, __logo__


def test_version() -> None:
    assert __version__
    assert isinstance(__version__, str)


def test_logo() -> None:
    assert __logo__ == "🧬"


def test_config_schema() -> None:
    from cytopert.config.schema import Config
    config = Config()
    assert config.workspace_path
    assert config.agents.defaults.model


def test_evidence_models() -> None:
    from cytopert.data.models import EvidenceEntry, EvidenceType, MechanismChain
    e = EvidenceEntry(id="e1", type=EvidenceType.DATA, summary="test")
    assert e.id == "e1"
    assert e.type == EvidenceType.DATA
    c = MechanismChain(id="c1", summary="chain", evidence_ids=["e1"])
    assert c.evidence_ids == ["e1"]


def test_tool_registry() -> None:
    from cytopert.agent.tools.registry import ToolRegistry
    from cytopert.agent.tools.evidence import EvidenceTool
    reg = ToolRegistry()
    reg.register(EvidenceTool())
    assert "evidence" in reg.tool_names
    assert len(reg.get_definitions()) == 1

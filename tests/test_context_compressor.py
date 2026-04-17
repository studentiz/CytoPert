"""Stage 4.C tests for the default ContextCompressor engine."""

from __future__ import annotations

from typing import Any

from cytopert.agent.context_compressor import CytoPertCompressor
from cytopert.providers.base import LLMResponse


class _StubProvider:
    def __init__(self, content: str | None) -> None:
        self.calls = 0
        self._content = content

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_base: str | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if self._content is None:
            return LLMResponse(content=None, finish_reason="error")
        return LLMResponse(content=self._content, finish_reason="stop", usage={})

    def get_default_model(self) -> str:
        return "stub"


def _msgs(n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": "S"}]
    for i in range(n):
        out.append({"role": "user", "content": f"u{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


def test_should_compress_threshold() -> None:
    eng = CytoPertCompressor(provider=None, model=None, context_length=1000,
                             threshold_percent=0.5)
    assert not eng.should_compress(prompt_tokens=100)
    assert eng.should_compress(prompt_tokens=600)
    eng.update_from_response({"prompt_tokens": 700, "completion_tokens": 0,
                              "total_tokens": 700})
    assert eng.should_compress()


def test_no_provider_falls_back_to_unchanged_messages() -> None:
    eng = CytoPertCompressor(provider=None, model=None,
                             protect_first_n=2, protect_last_n=2,
                             context_length=10)
    msgs = _msgs(8)
    out = eng.compress(msgs)
    assert out == msgs


def test_compression_protects_head_tail_and_evidence() -> None:
    prov = _StubProvider("SUMMARY OF THE MIDDLE")
    eng = CytoPertCompressor(
        provider=prov, model="stub",
        protect_first_n=2, protect_last_n=2, context_length=10,
    )
    eng.protect_evidence("tool_scanpy_de_keepme")
    msgs = _msgs(6)
    # Inject a tool result that mentions the protected id smack in the middle.
    msgs.insert(
        len(msgs) // 2,
        {"role": "tool", "content": "result tool_scanpy_de_keepme genes=[NFATC1]"},
    )
    out = eng.compress(msgs)
    # Head + summary message + protected tool + tail
    roles = [m["role"] for m in out]
    assert roles[:2] == ["system", "user"]
    assert roles[2] == "system"  # summary
    assert "Compressed summary" in out[2]["content"]
    assert any("tool_scanpy_de_keepme" in str(m.get("content", "")) for m in out)
    assert prov.calls == 1
    assert eng.compression_count == 1


def test_compression_skipped_when_below_protected_floor() -> None:
    prov = _StubProvider("never used")
    eng = CytoPertCompressor(
        provider=prov, model="stub",
        protect_first_n=3, protect_last_n=3, context_length=10,
    )
    msgs = _msgs(2)  # 5 messages total, below 6
    out = eng.compress(msgs)
    assert out == msgs
    assert prov.calls == 0


def test_compression_falls_back_when_summariser_errors() -> None:
    prov = _StubProvider(None)  # always returns finish_reason='error'
    eng = CytoPertCompressor(
        provider=prov, model="stub",
        protect_first_n=1, protect_last_n=1, context_length=10,
    )
    msgs = _msgs(5)
    out = eng.compress(msgs)
    assert out == msgs


def test_evidence_protection_helpers() -> None:
    eng = CytoPertCompressor(provider=None, model=None)
    eng.protect_evidence(["id_a", "id_b"])
    eng.protect_evidence("id_c")
    eng.protect_evidence("id_a")  # dedup
    assert eng.evidence_id_protect == ["id_a", "id_b", "id_c"]
    assert eng.is_protected({"role": "tool", "content": "saw id_b inside"})
    assert not eng.is_protected({"role": "user", "content": "no ids here"})


def test_get_status_contains_required_keys() -> None:
    eng = CytoPertCompressor(provider=None, model=None, context_length=100,
                             threshold_percent=0.7)
    eng.update_from_response({"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15})
    status = eng.get_status()
    for key in ("engine", "last_prompt_tokens", "threshold_tokens",
                "context_length", "usage_percent", "compression_count",
                "protected_evidence_ids"):
        assert key in status

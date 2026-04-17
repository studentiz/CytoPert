"""Stage 6.B tests for the Anthropic prompt-caching helper."""

from __future__ import annotations

from cytopert.providers.prompt_caching import (
    apply_anthropic_cache_control,
    is_anthropic_model,
)


def test_is_anthropic_model_known_families() -> None:
    assert is_anthropic_model("claude-sonnet-4")
    assert is_anthropic_model("claude-3-opus")
    assert is_anthropic_model("anthropic/claude-haiku")
    assert is_anthropic_model("openrouter/anthropic/claude-3.5-sonnet")
    assert is_anthropic_model("openrouter/claude-3-opus")
    assert not is_anthropic_model("deepseek/deepseek-chat")
    assert not is_anthropic_model("gpt-4o")
    assert not is_anthropic_model("")
    assert not is_anthropic_model(None)


def _has_marker(msg: dict) -> bool:
    if "cache_control" in msg:
        return True
    content = msg.get("content")
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        return "cache_control" in content[-1]
    return False


def test_marks_system_and_last_three_non_system() -> None:
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u1"},   # too old
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = apply_anthropic_cache_control(msgs)
    assert _has_marker(out[0]), out[0]
    assert _has_marker(out[2])
    assert _has_marker(out[3])
    assert _has_marker(out[4])
    assert not _has_marker(out[1])


def test_string_content_promoted_to_list() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_anthropic_cache_control(msgs)
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0]["text"] == "hi"
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_tool_role_only_marked_on_native_anthropic_path() -> None:
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "tool", "content": "T", "tool_call_id": "x"},
        {"role": "user", "content": "u2"},
    ]
    out_oai = apply_anthropic_cache_control(msgs, native_anthropic=False)
    tool_msg = next(m for m in out_oai if m["role"] == "tool")
    assert "cache_control" not in tool_msg
    out_native = apply_anthropic_cache_control(msgs, native_anthropic=True)
    tool_msg2 = next(m for m in out_native if m["role"] == "tool")
    assert "cache_control" in tool_msg2


def test_input_messages_are_not_mutated() -> None:
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    apply_anthropic_cache_control(msgs)
    assert msgs[0]["content"] == "S"  # still a string -> not promoted
    assert "cache_control" not in msgs[0]


def test_empty_messages_returned_unchanged() -> None:
    assert apply_anthropic_cache_control([]) == []

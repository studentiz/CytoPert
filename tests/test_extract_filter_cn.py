"""Stage 2.7 tests for AgentLoop._extract_filter Chinese-input compatibility.

The forced-tool-call parser intentionally accepts the most common CJK
punctuation tokens -- full-width colon (``\\uff1a``), full-width comma
(``\\uff0c``), and the verb meaning "set to" (``\\u8bbe\\u4e3a``) -- so
researchers writing in Chinese can drive ``census_query`` without
quoting around English ASCII syntax. The runtime achieves this with a
one-pass ASCII normalisation in
``AgentLoop._normalise_cn_input``; this test file exercises both the
ASCII output and the full-width input shapes so future refactors do not
silently regress the policy.
"""

from __future__ import annotations

import pytest

from cytopert.agent.loop import AgentLoop


@pytest.mark.parametrize(
    "text, key, expected",
    [
        ("obs_value_filter=tissue == 'blood'", "obs_value_filter", "tissue == 'blood'"),
        ("obs_value_filter:tissue == 'blood'", "obs_value_filter", "tissue == 'blood'"),
        ("obs_value_filter\uff1atissue == 'blood'", "obs_value_filter", "tissue == 'blood'"),
        ("obs_value_filter\u8bbe\u4e3atissue == 'blood'", "obs_value_filter", "tissue == 'blood'"),
        (
            "obs_value_filter=tissue == 'blood', max_cells=200",
            "obs_value_filter", "tissue == 'blood'",
        ),
        (
            "obs_value_filter=tissue == 'blood'\uff0c max_cells=200",
            "obs_value_filter", "tissue == 'blood'",
        ),
        ("census_version=2025-11-08", "census_version", "2025-11-08"),
        ("max_cells=2000", "max_cells", "2000"),
    ],
)
def test_extract_filter_recognises_known_keys(text: str, key: str, expected: str) -> None:
    assert AgentLoop._extract_filter(text, key) == expected


def test_extract_filter_returns_none_when_key_absent() -> None:
    assert AgentLoop._extract_filter("just chitchat", "obs_value_filter") is None

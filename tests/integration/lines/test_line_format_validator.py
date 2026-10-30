# ruff: noqa: N999 — line test module naming per plan's lines taxonomy.
"""Line test: ``validate_r2c2_answer`` format rules (pipeline gate).

Rules, in order:
  1. Empty / non-str → ``empty``.
  2. Exact ``CANONICAL_REFUSAL_TEXT`` match → pass (bypasses ';' check).
  3. Collapse internal \\n/\\r/\\t runs to a single space.
  4. Any sentinel fragment (``<tool_call``, ``<|``, …) → ``sentinel_leak``.
  5. Any ``;`` remaining → ``contains_semicolon``.
  6. Length > ``ANSWER_MAX_LEN`` → ``too_long`` (truncated in ``cleaned``).
"""

from __future__ import annotations

from r2c2_agent.agent.pipeline.answer_format import (
    ANSWER_MAX_LEN,
    CANONICAL_REFUSAL_TEXT,
    validate_r2c2_answer,
)


def test_empty_rejected() -> None:
    r = validate_r2c2_answer("")
    assert r.ok is False
    assert r.reason == "empty"


def test_non_str_rejected() -> None:
    r = validate_r2c2_answer(None)  # type: ignore[arg-type]
    assert r.ok is False
    assert r.reason == "empty"


def test_canonical_refusal_bypasses_semicolon_rule() -> None:
    """The canonical refusal text contains a ';' inside its prose, but
    it's an exact-match pass so the semicolon rule is bypassed."""
    assert ";" in CANONICAL_REFUSAL_TEXT
    r = validate_r2c2_answer(CANONICAL_REFUSAL_TEXT)
    assert r.ok is True
    assert r.cleaned == CANONICAL_REFUSAL_TEXT


def test_whitespace_collapse_normalises() -> None:
    r = validate_r2c2_answer("Luke\nSkywalker\tpilot")
    assert r.ok is True
    assert "\n" not in r.cleaned
    assert "\t" not in r.cleaned
    assert r.cleaned == "Luke Skywalker pilot"


def test_sentinel_leak_rejected() -> None:
    r = validate_r2c2_answer("Luke <tool_call>fake</tool_call>")
    assert r.ok is False
    assert r.reason == "sentinel_leak"


def test_semicolon_rejected_in_generic_answer() -> None:
    r = validate_r2c2_answer("Luke; Skywalker")
    assert r.ok is False
    assert r.reason == "contains_semicolon"


def test_too_long_rejected_and_truncated() -> None:
    long = "x" * (ANSWER_MAX_LEN + 50)
    r = validate_r2c2_answer(long)
    assert r.ok is False
    assert r.reason == "too_long"
    assert len(r.cleaned) == ANSWER_MAX_LEN


def test_happy_path_minimal_cleaned() -> None:
    r = validate_r2c2_answer("  taxi driver  ")
    assert r.ok is True
    # Strip whitespace around.
    assert r.cleaned == "taxi driver"

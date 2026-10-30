"""Guided-regex shape evaluator.

Checks whether a raw finalizer LLM output matches the
``FINALIZER_GUIDED_REGEX`` pattern
(``^(Answer: [^\\n;]{1,200}|UNKNOWN)$``). Used as a quality dimension
for the tier-3 / tier-4 node-quality report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FINALIZER_REGEX = re.compile(r"^(Answer: [^\n;]{1,200}|UNKNOWN)$")

__all__ = ["GuidedRegexScore", "evaluate_guided_regex"]


@dataclass(frozen=True)
class GuidedRegexScore:
    """Per-output regex compliance score."""

    matches: bool
    reason: str
    preview: str


def evaluate_guided_regex(raw_output: str) -> GuidedRegexScore:
    """Return whether ``raw_output`` satisfies the finalizer guided regex.

    Args:
        raw_output: Raw LLM content before any prefix stripping.

    Returns:
        ``GuidedRegexScore`` with ``matches`` True iff the regex admits
        the string verbatim. The reason field gives a short tag
        (``ok`` / ``missing_prefix`` / ``too_long`` / ``contains_newline`` /
        ``contains_semicolon`` / ``empty``). Preview is truncated to 60
        chars for table rendering.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return GuidedRegexScore(False, "empty", "")
    s = raw_output.strip()
    preview = s[:60].replace("\n", " ")
    if _FINALIZER_REGEX.match(s):
        return GuidedRegexScore(True, "ok", preview)
    # Diagnose why it failed.
    if "\n" in s or "\r" in s:
        return GuidedRegexScore(False, "contains_newline", preview)
    if ";" in s:
        return GuidedRegexScore(False, "contains_semicolon", preview)
    if s == "UNKNOWN":
        return GuidedRegexScore(True, "ok", preview)
    if not s.startswith("Answer: "):
        return GuidedRegexScore(False, "missing_prefix", preview)
    if len(s) - len("Answer: ") > 200:
        return GuidedRegexScore(False, "too_long", preview)
    return GuidedRegexScore(False, "unknown_shape", preview)

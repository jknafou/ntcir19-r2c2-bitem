"""Shared R2C2 character-set and length primitives.

Two validators import these:

* ``r2c2_agent.agent.pipeline.answer_format`` checks each answer inside the
  pipeline, when a tier emits it.
* ``r2c2_agent.agent.submission_validator`` checks each PR and AC line once
  the files are formatted.

The rules derive from the R2C2 submission format. The two validators work
at different granularities but must agree on the primitives: which
whitespace is banned, what counts as a sentinel leak, how semicolons are
treated.
"""

from __future__ import annotations

import re

__all__ = [
    "ANSWER_MAX_LEN",
    "AC_FIELD_SEPARATOR",
    "BANNED_WHITESPACE_RE",
    "CANONICAL_REFUSAL_TEXT",
    "SENTINEL_FRAGMENTS",
]


# Answer-string maximum length. Matches the 200-char ceiling applied at
# every tier-4 emission site in pipeline_handlers.py. The R2C2 task does
# not mandate a cap; it describes answers as "a word, phrase, sentence,
# quotation from a passage, etc." A 200-char sentence is ample for every
# question type in the corpus.
ANSWER_MAX_LEN = 200


# AC file field separator: ``AnswerString;ConfidenceScore`` on one line,
# ``NuggetNum;PRrunName;PassageRank;NuggetText`` on each nugget line. An
# answer containing this character would corrupt the AC file.
AC_FIELD_SEPARATOR = ";"


# Canonical refusal string emitted by every tier-5 path. The phrasing
# matches the ``REFUSAL_REGEX`` trigger words ("insufficient"/"cannot"/"no
# evidence"/"unable"/"refuse") so ``validate_finish(low_confidence_refusal)``
# accepts it. It contains one ``;`` inside prose, which
# ``validate_r2c2_answer`` allows by exact-string match because that ``;``
# is not a field separator.
# ``pipeline_handlers.CANONICAL_REFUSAL_ANSWER`` re-exports this value.
CANONICAL_REFUSAL_TEXT = (
    "The corpus does not support an answer to this question; "
    "insufficient evidence to respond."
)


# Fragments showing that the model emitted tool-call JSON, a chat-template
# marker or similar scaffolding in place of an answer. An answer containing
# one is rejected. Matched case-insensitively.
SENTINEL_FRAGMENTS: tuple[str, ...] = (
    "<tool_call",
    "</tool_call",
    "<|",
    "```json",
)


# Whitespace banned inside an emitted answer. The AC file puts each
# answer on one line as ``answer;confidence``, so ``\n``/``\r`` would split
# it across physical lines, and ``\t`` would break a tab-delimited export.
# The pattern matches runs, so substitution leaves one space per run.
BANNED_WHITESPACE_RE: re.Pattern[str] = re.compile(r"[\n\r\t]+")

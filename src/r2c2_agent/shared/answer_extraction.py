"""Robust answer and confidence extraction from LLM responses.

Strips markdown formatting using markdown-it-py, then extracts the
answer from the last "Answer: ... Confidence: N" pair.

Used by:
- src/r2c2_agent/agent/loop.py (runtime extraction)
- src/r2c2_agent/orchestrator/worker.py (post-trace draft extraction)
- src/r2c2_agent/orchestrator/generate_submission.py (passage + nugget helpers)
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from markdown_it import MarkdownIt

__all__ = [
    "extract_answer_and_confidence",
    "strip_markdown",
]

# Markdown stripping

_MD = MarkdownIt()


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def get_text(self) -> str:
        return "".join(self.text)


# Patterns that markdown-it doesn't catch (e.g. "** text" with leading **)
_LEADING_STARS_RE = re.compile(r"^\*{1,3}\s+")
_STRAY_STARS_RE = re.compile(r"\*{1,3}")


def strip_markdown(text: str) -> str:
    """Remove markdown formatting from text, returning plain text.

    Uses markdown-it-py to parse markdown to HTML, then strips HTML tags.
    Also handles edge cases that markdown-it doesn't process (like
    unmatched ** at the start of a line).

    Args:
        text: Text potentially containing markdown formatting.

    Returns:
        Plain text with markdown removed.
    """
    if not text:
        return text
    html = _MD.render(text)
    stripper = _HTMLStripper()
    stripper.feed(html)
    result = stripper.get_text().strip()
    # markdown-it leaves unmatched bold markers in place.
    result = _LEADING_STARS_RE.sub("", result)
    if "*" in result:
        result = _STRAY_STARS_RE.sub("", result).strip()
    return result


# Answer + Confidence extraction

_ANSWER_RE = re.compile(r"[Aa]nswer:")
_CONFIDENCE_RE = re.compile(r"[Cc]onfidence:\s*(\d+)")


def extract_answer_and_confidence(content: str) -> tuple[str, int]:
    """Extract answer and confidence from LLM response content.

    Algorithm:
    1. Strip markdown from the entire content
    2. Find all "Answer:" occurrences
    3. Take the last one
    4. Extract everything from after that "Answer:" until "Confidence:"
    5. Strip whitespace from the answer
    6. Parse confidence as int
    7. If answer is empty, return the full stripped content as fallback

    Args:
        content: Full content of the last assistant message.

    Returns:
        Tuple of (answer_string, confidence_score).
        Returns ("", 50) if nothing found.
    """
    if not content:
        return "", 50

    clean = strip_markdown(content)

    # The last "Answer:" is the one the model settled on.
    matches = list(_ANSWER_RE.finditer(clean))
    if not matches:
        return "", 50

    last_answer = matches[-1]
    after_answer = clean[last_answer.end() :]

    conf_match = _CONFIDENCE_RE.search(after_answer)
    if conf_match:
        answer_text = after_answer[: conf_match.start()].strip()
        confidence = int(conf_match.group(1))
    else:
        # No "Confidence:" follows: take the rest of the text.
        answer_text = after_answer.strip()
        confidence = 50

    answer_text = answer_text.strip().strip("\n").strip()

    # Empty or multi-line answers are returned as they are: the agentic
    # loop (see loop.py) asks the LLM to reformulate them.

    return answer_text, min(100, max(0, confidence))

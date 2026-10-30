"""R2C2 submission format validator.

Validates PR and AC file content against the R2C2 submission format.
Works at the text level (after formatting), complementing
``postprocess.validate_submission`` which works at the dataclass level
(before formatting).

For the per-answer validator invoked inside the pipeline at every tier
emission site, before any post-processing, see
``r2c2_agent.agent.pipeline.answer_format.validate_r2c2_answer``.
Both validators draw their char-set rules from
``r2c2_agent.agent.pipeline._r2c2_rules``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "ValidationReport",
    "validate_ac_file",
    "validate_pr_file",
]


@dataclass
class ValidationReport:
    """Aggregated validation result for a PR+AC submission pair."""

    pr_errors: list[str] = field(default_factory=list)
    ac_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if both files pass all checks."""
        return not self.pr_errors and not self.ac_errors

    def summary(self) -> str:
        """Human-readable summary string."""
        if self.is_valid:
            return "VALID: No errors found."
        parts: list[str] = []
        if self.pr_errors:
            parts.append(f"PR errors ({len(self.pr_errors)}):")
            parts.extend(f"  - {e}" for e in self.pr_errors)
        if self.ac_errors:
            parts.append(f"AC errors ({len(self.ac_errors)}):")
            parts.extend(f"  - {e}" for e in self.ac_errors)
        return "\n".join(parts)


# PR file validation

_OPEN_TAG_RE = re.compile(r"^<([^/][^>]*)>$")
_CLOSE_TAG_RE = re.compile(r"^</([^>]+)>$")


def validate_pr_file(pr_text: str, run_name: str = "") -> list[str]:
    """Validate PR file content against R2C2 spec.

    Checks per line:
      - Exactly 4 semicolon-separated fields (maxsplit=3)
      - QuestionID non-empty
      - PassageRank parses as positive integer
      - DocID non-empty
      - PassageText non-empty, no embedded newlines

    Checks per question:
      - At most 20 passages
      - Ranks are consecutive 1..n with no ties

    Args:
        pr_text: Full PR file content.
        run_name: PR run name (for future cross-file checks).

    Returns:
        List of error strings. Empty list means valid.
    """
    _ = run_name  # reserved for future PR naming checks
    errors: list[str] = []
    # question_id -> list of ranks seen
    question_ranks: dict[str, list[int]] = {}
    # question_id -> set of (doc_id, passage_text) tuples — a given
    # question must not list the same string-identical passage twice.
    question_passages: dict[str, set[tuple[str, str]]] = {}

    lines = pr_text.strip().split("\n") if pr_text.strip() else []

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split(";", 3)
        if len(parts) < 4:
            errors.append(
                f"PR line {line_num}: expected 4 semicolon-separated fields, "
                f"got {len(parts)}"
            )
            continue

        qid, rank_str, doc_id, passage_text = parts

        if not qid:
            errors.append(f"PR line {line_num}: empty QuestionID")

        try:
            rank = int(rank_str)
        except ValueError:
            errors.append(
                f"PR line {line_num}: PassageRank '{rank_str}' is not an integer"
            )
            continue

        if rank < 1 or rank > 20:
            errors.append(
                f"PR line {line_num}: PassageRank {rank} out of range [1, 20]"
            )

        if not doc_id:
            errors.append(f"PR line {line_num} ({qid}): empty DocID")

        if not passage_text:
            errors.append(f"PR line {line_num} ({qid}): empty PassageText")
        elif "\r" in passage_text or "\n" in passage_text:
            errors.append(
                f"PR line {line_num} ({qid}): PassageText contains line breaks"
            )
        elif "\t" in passage_text:
            errors.append(
                f"PR line {line_num} ({qid}): PassageText contains tab characters"
            )
        elif "  " in passage_text:
            errors.append(
                f"PR line {line_num} ({qid}): PassageText contains double spaces"
            )

        # Duplicate-passage check (doc_id + passage_text pair, per qid).
        # Corpus chunks occasionally produce two distinct passage_ids whose
        # doc_id + text render identically; the submission must carry each
        # passage only once so rank slots aren't wasted.
        pair = (doc_id, passage_text)
        if qid and pair in question_passages.get(qid, set()):
            errors.append(
                f"PR line {line_num} ({qid}): duplicate (doc_id, passage_text) "
                f"pair at rank {rank} — doc_id={doc_id!r}"
            )
        question_passages.setdefault(qid, set()).add(pair)

        question_ranks.setdefault(qid, []).append(rank)

    # Per-question checks
    for qid, ranks in question_ranks.items():
        if len(ranks) > 20:
            errors.append(f"PR {qid}: {len(ranks)} passages (max 20)")

        sorted_ranks = sorted(ranks)
        expected = list(range(1, len(ranks) + 1))
        if sorted_ranks != expected:
            if len(sorted_ranks) != len(set(sorted_ranks)):
                errors.append(f"PR {qid}: duplicate ranks detected")
            else:
                errors.append(
                    f"PR {qid}: ranks are not consecutive 1..{len(ranks)} "
                    f"(got {sorted_ranks})"
                )

    return errors


# AC file validation


def validate_ac_file(
    ac_text: str,
    run_name: str,
    *,
    pr_question_ids: set[str] | None = None,
    pr_rank_map: dict[str, set[int]] | None = None,
    require_nuggets: bool = False,
) -> list[str]:
    """Validate AC file content against R2C2 spec.

    State-machine parser:
      OUTSIDE → ``<QID>`` → INSIDE → ``</QID>`` → OUTSIDE

    Checks:
      - XML tags well-formed and balanced
      - Empty blocks allowed (unanswered questions)
      - Answer line: ``answer;confidence`` (rsplit on last ``;``)
      - Confidence is integer in [0, 100]
      - Nugget records: 4 fields ``NuggetNum;PRrunName;PassageRank;Text``
      - NuggetNum is integer in [1, 10], consecutive within block
      - At most 10 nuggets per block
      - PRrunName matches ``run_name`` if provided
      - PassageRank exists in PR if ``pr_rank_map`` provided

    Args:
        ac_text: Full AC file content.
        run_name: Expected PR run name in nugget records.
        pr_question_ids: Set of question IDs from PR file (for cross-check).
        pr_rank_map: Maps question_id → set of PR ranks (for cross-check).
        require_nuggets: If True, blocks with an answer but no nuggets
            are reported as errors.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []
    lines = ac_text.strip().split("\n") if ac_text.strip() else []

    current_qid: str | None = None
    seen_qids: list[str] = []
    block_line_idx = 0  # 0 = expecting answer line, 1+ = nugget lines
    nugget_nums: list[int] = []

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        open_match = _OPEN_TAG_RE.match(line)
        close_match = _CLOSE_TAG_RE.match(line)

        if open_match and current_qid is None:
            tag_name: str = open_match.group(1) or ""
            current_qid = tag_name
            seen_qids.append(tag_name)
            block_line_idx = 0
            nugget_nums = []
            continue

        if open_match and current_qid is not None:
            errors.append(
                f"AC line {line_num}: nested opening tag <{open_match.group(1)}> "
                f"inside block <{current_qid}>"
            )
            continue

        if close_match:
            tag_id = close_match.group(1)
            if current_qid is None:
                errors.append(
                    f"AC line {line_num}: closing tag </{tag_id}> without "
                    f"matching opening tag"
                )
                continue
            if tag_id != current_qid:
                errors.append(
                    f"AC line {line_num}: closing tag </{tag_id}> does not "
                    f"match opening tag <{current_qid}>"
                )

            _validate_nugget_nums(
                current_qid,
                nugget_nums,
                errors,
                require_nuggets=require_nuggets and block_line_idx > 0,
            )

            current_qid = None
            continue

        # Inside a block
        if current_qid is None:
            errors.append(
                f"AC line {line_num}: content outside of question block: {line[:50]}"
            )
            continue

        if block_line_idx == 0:
            # Answer line: AnswerString;ConfidenceScore
            _validate_answer_line(line, line_num, current_qid, errors)
            block_line_idx = 1
        else:
            _validate_nugget_line(
                line,
                line_num,
                current_qid,
                run_name,
                nugget_nums,
                pr_rank_map,
                errors,
            )
            block_line_idx += 1

    if current_qid is not None:
        errors.append(f"AC: unclosed block <{current_qid}> at end of file")

    # Cross-reference with PR question IDs
    if pr_question_ids is not None:
        ac_qids = set(seen_qids)
        missing_in_ac = pr_question_ids - ac_qids
        missing_in_pr = ac_qids - pr_question_ids
        for qid in sorted(missing_in_ac):
            errors.append(f"AC: question {qid} in PR but not in AC")
        for qid in sorted(missing_in_pr):
            errors.append(f"AC: question {qid} in AC but not in PR")

    return errors


def _validate_answer_line(
    line: str,
    line_num: int,
    qid: str,
    errors: list[str],
) -> None:
    """Validate the answer;confidence line of an AC block."""
    parts = line.rsplit(";", 1)
    if len(parts) < 2:
        errors.append(
            f"AC line {line_num} ({qid}): answer line missing semicolon "
            f"separator (expected 'answer;confidence')"
        )
        return

    answer, confidence_str = parts
    if not answer.strip():
        errors.append(f"AC line {line_num} ({qid}): empty answer string")
    elif "\n" in answer or "\r" in answer or "\t" in answer:
        errors.append(
            f"AC line {line_num} ({qid}): answer contains whitespace characters "
            f"(\\n, \\r, or \\t)"
        )

    try:
        confidence = int(confidence_str.strip())
    except ValueError:
        errors.append(
            f"AC line {line_num} ({qid}): confidence '{confidence_str.strip()}' "
            f"is not an integer"
        )
        return

    if confidence < 0 or confidence > 100:
        errors.append(
            f"AC line {line_num} ({qid}): confidence {confidence} out of range [0, 100]"
        )


def _validate_nugget_line(
    line: str,
    line_num: int,
    qid: str,
    run_name: str,
    nugget_nums: list[int],
    pr_rank_map: dict[str, set[int]] | None,
    errors: list[str],
) -> None:
    """Validate a single nugget record line."""
    parts = line.split(";", 3)
    if len(parts) < 4:
        errors.append(
            f"AC line {line_num} ({qid}): nugget record needs 4 fields "
            f"(NuggetNum;PRrunName;PassageRank;Text), got {len(parts)}"
        )
        return

    num_str, pr_run, rank_str, nugget_text = parts

    try:
        num = int(num_str.strip())
    except ValueError:
        errors.append(
            f"AC line {line_num} ({qid}): NuggetNum '{num_str.strip()}' "
            f"is not an integer"
        )
        return

    if num < 1 or num > 10:
        errors.append(
            f"AC line {line_num} ({qid}): NuggetNum {num} out of range [1, 10]"
        )

    nugget_nums.append(num)

    if run_name and pr_run.strip() != run_name:
        errors.append(
            f"AC line {line_num} ({qid}): PRrunName '{pr_run.strip()}' "
            f"does not match expected '{run_name}'"
        )

    try:
        rank = int(rank_str.strip())
    except ValueError:
        errors.append(
            f"AC line {line_num} ({qid}): PassageRank '{rank_str.strip()}' "
            f"is not an integer"
        )
        return

    if pr_rank_map is not None and qid in pr_rank_map:
        if rank not in pr_rank_map[qid]:
            errors.append(
                f"AC line {line_num} ({qid}): PassageRank {rank} "
                f"not found in PR entries"
            )

    if not nugget_text.strip():
        errors.append(f"AC line {line_num} ({qid}): empty nugget text")


def _validate_nugget_nums(
    qid: str,
    nugget_nums: list[int],
    errors: list[str],
    *,
    require_nuggets: bool = False,
) -> None:
    """Check that nugget numbers are consecutive 1..n within a block.

    Args:
        qid: Question ID for error messages.
        nugget_nums: List of nugget numbers seen in the block.
        errors: Accumulator for error strings.
        require_nuggets: If True, empty blocks (no nuggets) are errors.
    """
    if not nugget_nums:
        if require_nuggets:
            errors.append(
                f"AC {qid}: block has answer but no nugget lines "
                f"(expected at least 1 nugget record)"
            )
        return

    if len(nugget_nums) > 10:
        errors.append(f"AC {qid}: {len(nugget_nums)} nuggets (max 10)")

    expected = list(range(1, len(nugget_nums) + 1))
    if nugget_nums != expected:
        errors.append(
            f"AC {qid}: nugget numbers not consecutive 1..{len(nugget_nums)} "
            f"(got {nugget_nums})"
        )

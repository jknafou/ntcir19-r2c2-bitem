"""Validate R2C2 submission files (PR and AC) against the format specification.

Runs the 8 format checks of the R2C2 submission specification.

Usage:
    python -m r2c2_agent.validation.validate_submission \
        --pr <run-output>/BITEM-PG-1 \
        --ac <run-output>/BITEM-AC-1 \
        --pr-run-name BITEM-PG-1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

__all__: list[str] = []


def check_pr_semicolon_separated(lines: list[str]) -> tuple[bool, str]:
    """Check 1: PR lines are semicolon-separated with 4 fields."""
    errors = []
    for i, line in enumerate(lines, 1):
        fields = line.split(";")
        if len(fields) != 4:
            errors.append(f"  Line {i}: expected 4 fields, got {len(fields)}")
    if errors:
        return False, f"FAIL: {len(errors)} lines with wrong field count\n" + "\n".join(
            errors[:5]
        )
    return True, f"PASS: all {len(lines)} lines have 4 semicolon-separated fields"


def check_pr_passage_rank(lines: list[str]) -> tuple[bool, str]:
    """Check 2: PassageRank is integer [1-20], no ties per question."""
    errors = []
    ranks_by_question: dict[str, list[int]] = {}
    for i, line in enumerate(lines, 1):
        fields = line.split(";")
        if len(fields) < 2:
            continue
        qid = fields[0]
        try:
            rank = int(fields[1])
            if rank < 1 or rank > 20:
                errors.append(f"  Line {i}: rank {rank} out of range [1-20]")
            ranks_by_question.setdefault(qid, []).append(rank)
        except ValueError:
            errors.append(f"  Line {i}: non-integer rank '{fields[1]}'")

    for qid, ranks in ranks_by_question.items():
        if len(ranks) != len(set(ranks)):
            from collections import Counter

            dupes = {r: c for r, c in Counter(ranks).items() if c > 1}
            errors.append(f"  {qid}: duplicate ranks {dupes}")

    if errors:
        return False, "FAIL: rank issues\n" + "\n".join(errors[:5])
    return True, "PASS: all passage ranks are valid integers [1-20] with no ties"


def check_pr_max_passages(lines: list[str]) -> tuple[bool, str]:
    """Check 3: Max 20 passages per question."""
    counts: dict[str, int] = {}
    for line in lines:
        qid = line.split(";")[0]
        counts[qid] = counts.get(qid, 0) + 1
    violations = {qid: c for qid, c in counts.items() if c > 20}
    if violations:
        return False, f"FAIL: questions with >20 passages: {violations}"
    return (
        True,
        f"PASS: all questions have <= 20 passages (max: {max(counts.values()) if counts else 0})",
    )


def check_ac_xml_tags(content: str) -> tuple[bool, str]:
    """Check 4: AC has XML-style <QuestionID>...</QuestionID> tags."""
    open_tags = re.findall(r"<([^/][^>]*)>", content)
    close_tags = re.findall(r"</([^>]+)>", content)

    if not open_tags:
        return False, "FAIL: no XML-style question tags found"

    # Filter out non-question tags (opening tags that have matching closing)
    question_ids = []
    for tag in open_tags:
        if tag in close_tags:
            question_ids.append(tag)

    if not question_ids:
        return False, "FAIL: no matching open/close tag pairs found"

    unmatched = set(open_tags) - set(close_tags)
    if unmatched:
        return False, f"FAIL: unmatched open tags: {unmatched}"

    return True, f"PASS: {len(question_ids)} question blocks with matching XML tags"


def check_ac_answer_confidence(content: str) -> tuple[bool, str]:
    """Check 5: AnswerString;ConfidenceScore on first line of each block."""
    errors = []
    blocks = re.findall(r"<([^/][^>]*)>\n(.*?)\n</\1>", content, re.DOTALL)

    for qid, block_content in blocks:
        lines = block_content.strip().split("\n")
        if not lines:
            errors.append(f"  {qid}: empty block")
            continue
        first_line = lines[0]
        parts = first_line.split(";")
        if len(parts) < 2:
            errors.append(f"  {qid}: first line missing semicolon separator")
            continue
        try:
            conf = int(parts[-1])
            if conf < 0 or conf > 100:
                errors.append(f"  {qid}: confidence {conf} out of range [0-100]")
        except ValueError:
            errors.append(f"  {qid}: non-integer confidence '{parts[-1]}'")

    if errors:
        return False, "FAIL: answer/confidence issues\n" + "\n".join(errors[:5])
    return (
        True,
        f"PASS: all {len(blocks)} blocks have valid AnswerString;ConfidenceScore",
    )


def check_ac_nugget_format(content: str) -> tuple[bool, str]:
    """Check 6: NuggetNum;PRrunName;PassageRank;NuggetText (max 10 nuggets)."""
    errors = []
    blocks = re.findall(r"<([^/][^>]*)>\n(.*?)\n</\1>", content, re.DOTALL)

    for qid, block_content in blocks:
        lines = block_content.strip().split("\n")
        nugget_lines = lines[1:]  # Skip answer line
        if len(nugget_lines) > 10:
            errors.append(f"  {qid}: {len(nugget_lines)} nuggets (max 10)")
        for j, nline in enumerate(nugget_lines):
            parts = nline.split(";")
            if len(parts) < 4:
                errors.append(
                    f"  {qid} nugget {j + 1}: expected 4+ fields, got {len(parts)}"
                )

    if errors:
        return False, "FAIL: nugget format issues\n" + "\n".join(errors[:5])
    return True, "PASS: all nugget lines have valid format"


def check_ac_passage_refs(
    ac_content: str,
    pr_lines: list[str],
    pr_run_name: str,
) -> tuple[bool, str]:
    """Check 7: AC PassageKeys reference valid PR passages."""
    # Build set of valid (question_id, rank) pairs from PR
    valid_keys: set[tuple[str, int]] = set()
    for line in pr_lines:
        fields = line.split(";")
        if len(fields) >= 2:
            try:
                valid_keys.add((fields[0], int(fields[1])))
            except ValueError:
                pass

    errors = []
    blocks = re.findall(r"<([^/][^>]*)>\n(.*?)\n</\1>", ac_content, re.DOTALL)

    for qid, block_content in blocks:
        lines = block_content.strip().split("\n")
        for nline in lines[1:]:
            parts = nline.split(";")
            if len(parts) >= 3:
                run_name = parts[1]
                try:
                    rank = int(parts[2])
                    if run_name == pr_run_name and (qid, rank) not in valid_keys:
                        errors.append(f"  {qid}: references rank {rank} not in PR")
                except ValueError:
                    errors.append(f"  {qid}: non-integer rank in nugget '{parts[2]}'")

    if errors:
        return False, "FAIL: invalid passage references\n" + "\n".join(errors[:5])
    return True, "PASS: all AC passage references are valid"


def check_ac_nugget_num(content: str) -> tuple[bool, str]:
    """Check 8: NuggetNum is integer [1-10]."""
    errors = []
    blocks = re.findall(r"<([^/][^>]*)>\n(.*?)\n</\1>", content, re.DOTALL)

    for qid, block_content in blocks:
        lines = block_content.strip().split("\n")
        for nline in lines[1:]:
            parts = nline.split(";")
            if parts:
                try:
                    nnum = int(parts[0])
                    if nnum < 1 or nnum > 10:
                        errors.append(f"  {qid}: NuggetNum {nnum} out of range [1-10]")
                except ValueError:
                    errors.append(f"  {qid}: non-integer NuggetNum '{parts[0]}'")

    if errors:
        return False, "FAIL: NuggetNum issues\n" + "\n".join(errors[:5])
    return True, "PASS: all NuggetNums are integers in [1-10]"


def main() -> None:
    """Run all 8 validation checks."""
    parser = argparse.ArgumentParser(description="Validate R2C2 submission files")
    parser.add_argument("--pr", type=Path, required=True, help="PR submission file")
    parser.add_argument("--ac", type=Path, required=True, help="AC submission file")
    parser.add_argument(
        "--pr-run-name",
        type=str,
        default="BITEM-DRY-PG-1",
        help="PR run name for cross-reference check",
    )
    args = parser.parse_args()

    print("=== R2C2 Submission Validation ===")
    print(f"PR file: {args.pr}")
    print(f"AC file: {args.ac}")
    print(f"PR run name: {args.pr_run_name}")
    print()

    if not args.pr.exists():
        print(f"ERROR: PR file not found: {args.pr}")
        sys.exit(1)
    if not args.ac.exists():
        print(f"ERROR: AC file not found: {args.ac}")
        sys.exit(1)

    pr_text = args.pr.read_text()
    ac_text = args.ac.read_text()
    pr_lines = [l for l in pr_text.strip().split("\n") if l.strip()]

    print(f"PR lines: {len(pr_lines)}")
    print(f"AC length: {len(ac_text)} chars")
    print()

    checks = [
        ("1. PR semicolon-separated, 4 fields", check_pr_semicolon_separated(pr_lines)),
        ("2. PR PassageRank [1-20], no ties", check_pr_passage_rank(pr_lines)),
        ("3. PR max 20 passages/question", check_pr_max_passages(pr_lines)),
        ("4. AC XML-style question tags", check_ac_xml_tags(ac_text)),
        ("5. AC AnswerString;ConfidenceScore", check_ac_answer_confidence(ac_text)),
        ("6. AC nugget format (max 10)", check_ac_nugget_format(ac_text)),
        (
            "7. AC PassageKey references valid PR",
            check_ac_passage_refs(ac_text, pr_lines, args.pr_run_name),
        ),
        ("8. AC NuggetNum [1-10]", check_ac_nugget_num(ac_text)),
    ]

    passed = 0
    failed = 0
    for name, (ok, detail) in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        print(f"  {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n=== Result: {passed}/8 PASS, {failed}/8 FAIL ===")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

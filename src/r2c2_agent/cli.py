"""Unified CLI for the published r2c2_agent inference pipeline.

Subcommands:

* ``submit``        — drive orchestrator → workers → aggregator → ZIP
                      emission end-to-end on a JSONL of questions.
* ``reaggregate-ac`` — replay a frozen PR submission's traces through
                      the AC re-aggregator.
* ``llm-judge``     — run the AC-4 LLM-as-judge pass and emit
                      ``judge_decisions.jsonl``.

Each subcommand shells out to the underlying SLURM launcher
(``src/r2c2_agent/slurm/launch.sh``) for the real GPU run, then invokes the
in-process aggregator + validator chain on the produced trace files.

Usage::

    python -m r2c2_agent submit --questions Q.jsonl --output OUT/
    python -m r2c2_agent reaggregate-ac \\
        --pr-dir RUN/submission --traces-dir RUN --output OUT/

The ``submit`` subcommand exposes the same env-var overrides honoured by
``src/r2c2_agent/slurm/launch.sh`` (``MODEL``, ``ENSEMBLE``,
``ENSEMBLE_PER_ITER``, ``MAX_ADAPTIVE_ENSEMBLE``, etc.). See the
launcher header for the full list. For smoke-test runs, pass ``--demo``
to ``launch.sh`` (smaller model, K=2, dry-run question set).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

__all__ = ["main"]

log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser.

    Returns:
        ArgumentParser with the ``submit``, ``reaggregate-ac`` and
        ``llm-judge`` subparsers.
    """
    parser = argparse.ArgumentParser(
        prog="r2c2_agent",
        description=(
            "r2c2_agent — published agentic-RAG inference + AC "
            "re-aggregation CLI (T1→T2→T4→T5 educated-guess ladder)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    # submit
    p_sub = sub.add_parser(
        "submit",
        help="Run the orchestrator end-to-end on a JSONL of questions.",
    )
    p_sub.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to questions JSONL (one record per line, schema: "
        "``{question_id, question, ...}``).",
    )
    p_sub.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory; iter_K/test_traces.jsonl + submission/ ZIPs land here.",
    )
    p_sub.add_argument(
        "--ensemble",
        type=str,
        default="adaptive",
        help="``adaptive`` (default — canonical noon-submission mode: roll "
        "up to K=4 with early-stop on two consecutive low-confidence "
        "iters) or a fixed-K integer (e.g. ``4`` for fixed K=4, ``1`` "
        "for a single-pass run).",
    )
    p_sub.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "Qwen/Qwen3.5-122B-A10B-FP8"),
        help="HF model identifier passed to vLLM workers.",
    )
    p_sub.add_argument(
        "--mode",
        type=str,
        default="question",
        choices=["question"],
        help="``question`` runs the agentic loop on the input questions JSONL.",
    )
    p_sub.add_argument(
        "--run-name",
        type=str,
        default="BITEM-PG-1",
        help="Run name embedded in PR/AC files (default: BITEM-PG-1).",
    )
    p_sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch invocation and exit without submitting.",
    )

    # reaggregate-ac
    p_ac = sub.add_parser(
        "reaggregate-ac",
        help="Re-aggregate the AC submission from frozen PR + iter_*/traces.",
    )
    p_ac.add_argument(
        "--pr-dir",
        type=Path,
        required=True,
        help="Directory containing the frozen ``BITEM-PG-{1,2}`` PR files "
        "and the original ``BITEM-AC-{1,2}`` (read-only).",
    )
    p_ac.add_argument(
        "--traces-dir",
        type=Path,
        required=True,
        help="Run directory containing ``iter_*/test_traces.jsonl`` rollouts.",
    )
    p_ac.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory; ``BITEM-AC-{1,2}.v2`` + ``BITEM-AC.v2.zip`` land here.",
    )
    p_ac.add_argument(
        "--source-questions",
        type=Path,
        default=None,
        help="Optional source questions JSONL for AC source-coverage check.",
    )
    p_ac.add_argument(
        "--strategy",
        type=str,
        default="trace-redesign",
        choices=[
            "trace-redesign",
            "best-rollout",
            "plurality",
            "llm-judge",
            "max-conf",
            "conf-sum-cluster",
        ],
        help=(
            "AC re-aggregation strategy. ``max-conf`` and "
            "``conf-sum-cluster`` are the AC-2 / AC-3 designs shipped in "
            "the final submission (order-statistic variance reduction + "
            "evidence pooling). ``best-rollout`` and ``plurality`` are "
            "preserved for backward compatibility with earlier AC files."
        ),
    )
    p_ac.add_argument(
        "--judge-decisions",
        type=Path,
        default=None,
        help=(
            "Path to judge_decisions.jsonl from `r2c2_agent llm-judge`. "
            "Required iff --strategy llm-judge."
        ),
    )
    p_ac.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Output AC filename suffix (e.g. '2', '3', '4'). When set, "
            "the PG-2 output is written as 'BITEM-AC-<SUFFIX>' instead "
            "of 'BITEM-AC-2.v2'."
        ),
    )
    p_ac.add_argument(
        "--allow-incomplete-judge",
        action="store_true",
        default=False,
        help=(
            "Development-only escape hatch for --strategy llm-judge. By "
            "default, AC-4 emission hard-fails unless every qid has a "
            "real (non-parse_failed, non-deferred) LLM judge decision. "
            "Set this flag to permit incomplete decisions for local "
            "development. Never use for a shipped BITEM-AC-4."
        ),
    )

    # llm-judge
    p_judge = sub.add_parser(
        "llm-judge",
        help=(
            "Run the AC-4 LLM-as-judge pass over all qids and emit "
            "judge_decisions.jsonl. Required input for "
            "`reaggregate-ac --strategy llm-judge`."
        ),
    )
    p_judge.add_argument("--pr-dir", type=Path, required=True)
    p_judge.add_argument("--traces-dir", type=Path, required=True)
    p_judge.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Source questions JSONL (id + question text).",
    )
    p_judge.add_argument(
        "--vllm-endpoint",
        type=str,
        required=True,
        help="OpenAI-compatible base URL, e.g. http://host:11000/v1",
    )
    p_judge.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-122B-A10B-FP8",
    )
    p_judge.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for judge_decisions.jsonl",
    )
    p_judge.add_argument(
        "--qid-filter",
        type=str,
        default=None,
        help="Comma-separated qids for smoke testing.",
    )
    p_judge.add_argument("--concurrency", type=int, default=1)

    return parser


def _launch_orchestrator(args: argparse.Namespace) -> int:
    """Submit the orchestrator SLURM job for ``submit``.

    Args:
        args: Parsed argparse namespace from ``submit``.

    Returns:
        Process exit code from the underlying ``launch.sh`` invocation.
    """
    launcher = _PKG_ROOT / "slurm" / "launch.sh"
    if not launcher.exists():
        log.error("launch.sh not found at %s — aborting", launcher)
        return 2

    env = os.environ.copy()
    env["INPUT"] = str(args.questions.resolve())
    env["OUTPUT"] = str(
        (args.output / "test_traces.jsonl").resolve()
    )  # legacy filename
    env["MODE"] = args.mode
    env["MODEL"] = args.model
    env["ENSEMBLE"] = str(args.ensemble)
    # Per-iter dispatch fires for any multi-rollout run (adaptive or fixed K>1).
    if args.ensemble == "adaptive" or (
        str(args.ensemble).isdigit() and int(args.ensemble) > 1
    ):
        env["ENSEMBLE_PER_ITER"] = "1"

    args.output.mkdir(parents=True, exist_ok=True)

    cmd = ["bash", str(launcher)]
    log.info(
        "submit: launching orchestrator: %s (env: ENSEMBLE=%s MODE=%s)",
        cmd,
        env["ENSEMBLE"],
        env["MODE"],
    )
    if args.dry_run:
        sys.stdout.write(
            "[dry-run] would invoke: "
            + " ".join(
                f"{k}={v}"
                for k, v in env.items()
                if k
                in ("INPUT", "OUTPUT", "MODE", "MODEL", "ENSEMBLE", "ENSEMBLE_PER_ITER")
            )
            + " "
            + " ".join(cmd)
            + "\n"
        )
        return 0
    return subprocess.call(cmd, env=env)


def _run_aggregator_on_output(args: argparse.Namespace) -> int:
    """Run the ensemble aggregator on the produced traces and emit ZIPs.

    Produces both submission runs the R2C2 spec expects:

    - **PG-1** — single-pass baseline (``--first-iter-only`` over iter_0).
    - **PG-2** — full K=N ensemble (all iter_*/test_traces.jsonl).

    Then renames the per-run ``<run_name>.PR`` / ``<run_name>.AC`` files
    to the unsuffixed ``BITEM-PG-{1,2}`` / ``BITEM-AC-{1,2}`` filenames
    required by the organiser, and packs them at the ZIP root into
    ``submission/BITEM-PR.zip`` + ``submission/BITEM-AC.zip``.

    Args:
        args: Parsed argparse namespace from ``submit``.

    Returns:
        Exit code from the aggregator (0 = success).
    """
    from r2c2_agent.orchestrator.ensemble_aggregator import main as agg_main

    base_run_name = args.run_name
    # Strip any trailing -N suffix from the supplied run name: both -1
    # (single-pass) and -2 (full ensemble) are emitted below.
    base_run_name = base_run_name.rstrip("12").rstrip("-")

    pg1_name = f"{base_run_name}-1"
    pg2_name = f"{base_run_name}-2"

    common_args = [
        "--run-dir",
        str(args.output.resolve()),
        "--source-questions",
        str(args.questions.resolve()),
    ]

    log.info("submit: aggregating PG-1 (single-pass / iter_0 only)")
    rc = agg_main([*common_args, "--run-name", pg1_name, "--first-iter-only"])
    if rc != 0:
        return rc

    log.info("submit: aggregating PG-2 (full ensemble)")
    rc = agg_main([*common_args, "--run-name", pg2_name])
    if rc != 0:
        return rc

    return _pack_submission_zips(args.output.resolve(), pg1_name, pg2_name)


def _pack_submission_zips(run_dir: Path, pg1_name: str, pg2_name: str) -> int:
    """Rename + zip the aggregator outputs per R2C2 spec.

    Layout produced::

        <run_dir>/
            <pg1_name>.PR        # produced by aggregator (PG-1 single-pass)
            <pg1_name>.AC
            <pg2_name>.PR        # produced by aggregator (PG-2 ensemble)
            <pg2_name>.AC
            submission/
                BITEM-PG-1
                BITEM-PG-2
                BITEM-AC-1
                BITEM-AC-2
                BITEM-PR.zip     ← BITEM-PG-{1,2} at zip root
                BITEM-AC.zip     ← BITEM-AC-{1,2} at zip root

    Per R2C2 organiser instructions
    (https://sakailab.com/r2c2-run-submission-instructions/), the ZIP
    files must contain the unsuffixed ``BITEM-PG-{1,2}`` /
    ``BITEM-AC-{1,2}`` filenames at the ZIP root with no path prefix.
    """
    import zipfile

    sub_dir = run_dir / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)

    pg1_pr = run_dir / f"{pg1_name}.PR"
    pg1_ac = run_dir / f"{pg1_name}.AC"
    pg2_pr = run_dir / f"{pg2_name}.PR"
    pg2_ac = run_dir / f"{pg2_name}.AC"

    for src in (pg1_pr, pg1_ac, pg2_pr, pg2_ac):
        if not src.exists():
            log.error("submission file missing: %s", src)
            return 1

    # Spec: AC filenames use ``AC`` middle, not ``PG`` (organiser quirk).
    ac1_name = pg1_name.replace("-PG-", "-AC-")
    ac2_name = pg2_name.replace("-PG-", "-AC-")

    pg1_target = sub_dir / pg1_name
    pg2_target = sub_dir / pg2_name
    ac1_target = sub_dir / ac1_name
    ac2_target = sub_dir / ac2_name
    pg1_target.write_bytes(pg1_pr.read_bytes())
    pg2_target.write_bytes(pg2_pr.read_bytes())
    ac1_target.write_bytes(pg1_ac.read_bytes())
    ac2_target.write_bytes(pg2_ac.read_bytes())

    pr_zip = sub_dir / "BITEM-PR.zip"
    ac_zip = sub_dir / "BITEM-AC.zip"
    with zipfile.ZipFile(pr_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pg1_target, arcname=pg1_name)
        zf.write(pg2_target, arcname=pg2_name)
    with zipfile.ZipFile(ac_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(ac1_target, arcname=ac1_name)
        zf.write(ac2_target, arcname=ac2_name)

    log.info("submission ZIPs ready:")
    log.info("  %s", pr_zip)
    log.info("  %s", ac_zip)
    return 0


def _run_reaggregate_ac(args: argparse.Namespace) -> int:
    """Drive the ``ac_reaggregator`` AC re-aggregation.

    Args:
        args: Parsed argparse namespace from ``reaggregate-ac``.

    Returns:
        Exit code from the re-aggregator (0 = success).
    """
    from r2c2_agent.orchestrator.ac_reaggregator import main as reagg_main

    reagg_args: list[str] = [
        "--pr-dir",
        str(args.pr_dir.resolve()),
        "--traces-dir",
        str(args.traces_dir.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    if args.source_questions is not None:
        reagg_args.extend(["--source-questions", str(args.source_questions.resolve())])
    strategy = getattr(args, "strategy", None)
    if strategy:
        reagg_args.extend(["--strategy", strategy])
    judge_decisions = getattr(args, "judge_decisions", None)
    if judge_decisions is not None:
        reagg_args.extend(["--judge-decisions", str(judge_decisions.resolve())])
    output_suffix = getattr(args, "output_suffix", None)
    if output_suffix:
        reagg_args.extend(["--output-suffix", output_suffix])
    if getattr(args, "allow_incomplete_judge", False):
        reagg_args.append("--allow-incomplete-judge")
    log.info("reaggregate-ac: invoking ac_reaggregator with %s", reagg_args)
    return reagg_main(reagg_args)


def _run_llm_judge(args: argparse.Namespace) -> int:
    """Drive the AC-4 LLM-as-judge pass.

    Args:
        args: Parsed argparse namespace from ``llm-judge``.

    Returns:
        Exit code from ``llm_judge.run_judge_pass`` (0 = success).
    """
    from r2c2_agent.orchestrator.llm_judge import main as judge_main

    judge_argv: list[str] = [
        "--pr-dir",
        str(args.pr_dir.resolve()),
        "--traces-dir",
        str(args.traces_dir.resolve()),
        "--questions",
        str(args.questions.resolve()),
        "--vllm-endpoint",
        args.vllm_endpoint,
        "--model",
        args.model,
        "--output",
        str(args.output.resolve()),
        "--concurrency",
        str(args.concurrency),
    ]
    if args.qid_filter:
        judge_argv.extend(["--qid-filter", args.qid_filter])
    log.info("llm-judge: invoking llm_judge with %s", judge_argv)
    return judge_main(judge_argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code. ``0`` on success, non-zero on any failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "submit":
        rc = _launch_orchestrator(args)
        if rc != 0 or args.dry_run:
            return rc
        return _run_aggregator_on_output(args)
    if args.cmd == "reaggregate-ac":
        return _run_reaggregate_ac(args)
    if args.cmd == "llm-judge":
        return _run_llm_judge(args)
    parser.error(f"unknown subcommand: {args.cmd}")
    return 2  # unreachable but keeps mypy happy

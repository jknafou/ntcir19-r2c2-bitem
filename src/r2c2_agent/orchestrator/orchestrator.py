#!/usr/bin/env python3
"""Self-managing distributed orchestrator for question rollout generation.

Submits GPU worker jobs across Baobab and Bamboo, monitors their lifecycle,
distributes work via HTTP, resubmits failed workers, and cancels everything
when done.

Usage:
    python -m r2c2_agent submit \\
        --questions data/r2c2/test_questions.jsonl \\
        --output data/runs/<run-name>/ \\
        --ensemble 4 \\
        --model Qwen/Qwen3.5-122B-A10B-FP8 \\
        --run-name BITEM-PG-2 \\
        --mode question
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

__all__: list[str] = []

log = logging.getLogger(__name__)

# Adaptive ensemble. ``--ensemble`` without a value sets args.ensemble to
# ADAPTIVE_SENTINEL; per-iteration dispatch keeps issuing rollouts until two
# consecutive records carry raw_confidence < ADAPTIVE_LOW_CONF_THRESHOLD
# (after at least MIN_ADAPTIVE_ITERATIONS) or until the MAX_ADAPTIVE_ENSEMBLE
# cap.
ADAPTIVE_SENTINEL = -1
MIN_ADAPTIVE_ITERATIONS = 3
ADAPTIVE_LOW_CONF_THRESHOLD = 50  # raw_confidence in 0..100
MAX_ADAPTIVE_ENSEMBLE = 4  # a fourth rollout gathers more passages


def adaptive_should_stop(fresh_confs: list[int]) -> tuple[bool, str | None]:
    """Pure predicate for the runtime adaptive-ensemble stop decision.

    Mirrors the inline check in ``process_item`` so tests can exercise
    the decision without standing up a full dispatch loop. Only
    *freshly dispatched* confidence values should be passed in — records
    replayed from disk on resume are intentionally excluded from the
    runtime predicate (they may be ghost-zeros from partial writes).

    Args:
        fresh_confs: raw_confidence values (0..100) for iterations
            freshly dispatched in the current session, in order.

    Returns:
        ``(True, reason)`` if the loop should break out early.
        ``(False, None)`` if the loop should continue to the next K
        (caller still enforces the ``MAX_ADAPTIVE_ENSEMBLE`` cap).

    Notes:
        - Never stops before ``MIN_ADAPTIVE_ITERATIONS`` fresh dispatches.
        - Stops only when the *last two* fresh confs are both strictly
          below ``ADAPTIVE_LOW_CONF_THRESHOLD``. A single low confidence
          is never enough; earlier low values flanked by a high one do
          not trigger either.
    """
    if len(fresh_confs) < MIN_ADAPTIVE_ITERATIONS:
        return False, None
    if len(fresh_confs) < 2:
        return False, None
    if (
        fresh_confs[-1] < ADAPTIVE_LOW_CONF_THRESHOLD
        and fresh_confs[-2] < ADAPTIVE_LOW_CONF_THRESHOLD
    ):
        return True, "two_consecutive_low_conf"
    return False, None


def _effective_ensemble_n(ensemble_arg: int) -> int:
    """Translate ``args.ensemble`` into an effective iteration count.

    The argparse sentinel ``ADAPTIVE_SENTINEL == -1`` is a runtime
    mode flag, not a literal iteration count. Any code that loops up
    to ``args.ensemble`` (resume scans, placeholder fills, per-iter
    cleanup) goes through this helper, because ``range(-1)`` is an
    empty sequence and the loop's work would be skipped silently.

    Args:
        ensemble_arg: the raw ``args.ensemble`` integer.

    Returns:
        ``MAX_ADAPTIVE_ENSEMBLE`` when adaptive mode is active,
        ``ensemble_arg`` otherwise.
    """
    if ensemble_arg == ADAPTIVE_SENTINEL:
        return MAX_ADAPTIVE_ENSEMBLE
    return ensemble_arg


async def write_resume_skipped_placeholders(
    *,
    item: dict[str, Any],
    rollout_idx: int,
    writer: Any,
    output_path: Path,
    ensemble_n: int,
    temperature: float,
    adaptive: bool = False,
) -> int:
    """Write ``termination="resume_skipped"`` records for every iter_K
    slot in ``[0, ensemble_n)`` that doesn't already hold a record for
    ``(question_id, rollout_idx)``.

    Used by the orchestrator's per-item retry-cap path: when a qid's
    retry budget is exhausted mid-ensemble, this closes out the
    remaining K-slots so ``load_done_ids`` treats the qid as
    terminally handled (max-cap rule in adaptive; full N in fixed-N).
    Without this, the qid is neither "done" nor "in the retry queue"
    and silently disappears from the output stream.

    Callers in adaptive mode pass ``ensemble_n =
    _effective_ensemble_n(args.ensemble)`` so the MAX_ADAPTIVE_ENSEMBLE
    cap is used instead of the raw ``-1`` sentinel. Fixed-N mode
    passes the fixed N directly.

    Args:
        item: the original work item (carries question_id/text/type).
        rollout_idx: outer rollout index (from rollouts-per-item loop).
        writer: async callable ``writer(record, dest=Path) -> None``
            that appends ``record`` as a JSONL line to ``dest``.
        output_path: the flat/base output path; per-iter destinations
            are resolved via :func:`_iter_output_path`.
        ensemble_n: effective iteration count (see
            :func:`_effective_ensemble_n`).
        temperature: sampling temperature to stamp on placeholders so
            downstream consumers can distinguish runs.
        adaptive: True when the caller is in adaptive mode. Sets
            ``adaptive_ensemble=True`` on each placeholder so the
            output-schema invariant (every record in an adaptive run
            carries ``adaptive_ensemble=True``) is preserved by the
            retry-cap fallback. Ignored in fixed-N mode.

    Returns:
        Number of placeholder records actually written (0 if all
        iter_K slots were already populated).
    """
    qid = str(item.get("question_id") or item.get("qid") or "")
    qtext = str(item.get("question_text") or "")
    qtype = str(item.get("question_type") or "")
    written = 0
    for k in range(ensemble_n):
        dest = _iter_output_path(output_path, k)
        if dest.exists():
            # Fast check: does this iteration already have a record
            # for this (qid, rollout) pair? Skip if yes.
            already_done = False
            try:
                with open(dest) as f_check:
                    for line in f_check:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if (
                                rec.get("question_id") == qid
                                and rec.get("rollout_idx", 0) == rollout_idx
                            ):
                                already_done = True
                                break
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass
            if already_done:
                continue
        placeholder: dict[str, Any] = {
            "question_id": qid,
            "question_text": qtext,
            "question_type": qtype,
            "rollout_idx": rollout_idx,
            "ensemble_idx": k,
            "ensemble_n": ensemble_n,
            "ensemble_excluded_pids": [],
            "ensemble_touched_delta": [],
            "temperature": temperature,
            "final_answer": "",
            "confidence": 0.0,
            "raw_confidence": 0,
            "termination": "resume_skipped",
            "iterations": 0,
            "tool_calls": [],
            "elapsed_seconds": 0.0,
        }
        if adaptive:
            placeholder["adaptive_ensemble"] = True
        await writer(placeholder, dest=dest)
        written += 1
    return written


# Primary/secondary cluster detection
#
# The orchestrator can be launched from either cluster, so that a run is
# still possible while one of them is in maintenance (the other is then the
# secondary, or absent entirely). Two constants drive every cross-cluster
# decision: the ``sbatch -M`` flag, the rsync direction and the login-node
# ssh hops.
#
#   PRIMARY_CLUSTER   — cluster the orchestrator is running on
#   SECONDARY_CLUSTER — the other cluster (rsync target, cross-cluster sbatch)
#
# Detection order:
#   1. Explicit override via env var PRIMARY_CLUSTER (pins CI/tests).
#   2. Hostname match on FQDN / nodename. Bamboo compute+login nodes have
#      "bamboo" in their FQDN; Baobab matches "baobab".
#   3. Fallback: "baobab".
#
# The GPU_POOL dict below keys on physical cluster names
# ("baobab"/"bamboo") because those refer to where GPUs actually live, not
# to the orchestrator's own location. PRIMARY_CLUSTER only influences the
# control-flow decisions (where to submit from, which direction to rsync).


def _detect_primary_cluster() -> str:
    """Detect which cluster the orchestrator is running on.

    Returns:
        Lower-case cluster name: ``"baobab"`` or ``"bamboo"``.
    """
    override = os.environ.get("PRIMARY_CLUSTER", "").strip().lower()
    if override in ("baobab", "bamboo"):
        return override
    candidates: list[str] = []
    try:
        candidates.append(os.uname().nodename)
    except OSError:
        pass
    try:
        candidates.append(socket.getfqdn())
    except OSError:
        pass
    for name in candidates:
        low = name.lower()
        if "bamboo" in low:
            return "bamboo"
        if "baobab" in low:
            return "baobab"
    return "baobab"  # default when no hostname matches


PRIMARY_CLUSTER: str = _detect_primary_cluster()
SECONDARY_CLUSTER: str = "baobab" if PRIMARY_CLUSTER == "bamboo" else "bamboo"

PROJECT = Path(__file__).resolve().parents[3]  # repo root
WORKER_SBATCH = "src/r2c2_agent/slurm/worker.sbatch"
ENDPOINTS_DIR = PROJECT / "src/r2c2_agent/slurm/endpoints"
ORCHESTRATOR_LOCK = ENDPOINTS_DIR / "orchestrator.lock"
PORT_REGISTRY_FILE = ENDPOINTS_DIR / "port_registry.json"
MODEL_DEFAULT = "Qwen/Qwen3.5-122B-A10B-FP8"

# Pool management tunables.
POOL_COOLDOWN_BASE = 30.0  # initial cooldown (seconds)
POOL_COOLDOWN_MAX = 600.0  # max cooldown (10 min)
POOL_MAX_CONSECUTIVE_FAILURES = (
    50  # stop resubmitting after this many consecutive failures
)
POOL_UNSUSPEND_AFTER = 1800.0  # auto-unsuspend after 30 min
POOL_STATUS_INTERVAL = 120.0  # log pool status every 2 min

# Deterministic tunnel port assignment
#
# Tunnel ports are a pure function of (cluster, job_id). This eliminates every
# form of port-registry drift: a worker computes its port from its own
# SLURM_JOB_ID; the orchestrator computes the same port for adoption. No state
# file, no counter, no races between orchestrator restarts.
#
# Scheme: port = BASE + (job_id % SPAN), range [BASE, BASE + SPAN).
# Collision risk for k live jobs in SPAN slots is ~k^2/(2*SPAN).
# For k=20 and PORT_SPAN=1000 that's ~0.2%. We linearly probe within the
# orchestrator when a collision is observed against the live-job snapshot and
# pass the resulting TUNNEL_PORT explicitly to sbatch.

TUNNEL_PORT_BASE = 19000
TUNNEL_PORT_SPAN = 1000  # range 19000-19999


def deterministic_tunnel_port(job_id: str | int) -> int:
    """Return the canonical tunnel port for a Bamboo job.

    Args:
        job_id: SLURM job ID (numeric string or int).

    Returns:
        Port number in [TUNNEL_PORT_BASE, TUNNEL_PORT_BASE + TUNNEL_PORT_SPAN).
    """
    try:
        j = int(str(job_id).strip())
    except (TypeError, ValueError):
        return TUNNEL_PORT_BASE
    return TUNNEL_PORT_BASE + (j % TUNNEL_PORT_SPAN)


def load_port_registry() -> dict[str, dict[str, Any]]:
    """Load the persistent job_id → {host, port} registry from disk.

    Also accepts the older format in which a value is a bare port int
    (Bamboo only).
    """
    if not PORT_REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(PORT_REGISTRY_FILE.read_text())
        result: dict[str, dict[str, Any]] = {}
        for k, v in data.items():
            if isinstance(v, int):
                # Bare port: assume localhost (Bamboo tunnel)
                result[str(k)] = {"host": "localhost", "port": int(v)}
            elif isinstance(v, dict) and "host" in v and "port" in v:
                result[str(k)] = {"host": str(v["host"]), "port": int(v["port"])}
        return result
    except Exception as e:
        log.warning("Failed to load port registry: %s", e)
        return {}


def save_port_registry(registry: dict[str, dict[str, Any]]) -> None:
    """Write the job_id → {host, port} mapping to disk atomically."""
    PORT_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PORT_REGISTRY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, indent=2, sort_keys=True))
    tmp.replace(PORT_REGISTRY_FILE)


def prune_port_registry(
    registry: dict[str, dict[str, Any]],
    live_job_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Remove entries for jobs that are no longer in squeue."""
    pruned = {jid: v for jid, v in registry.items() if jid in live_job_ids}
    dropped = len(registry) - len(pruned)
    if dropped > 0:
        print(
            f"[port-registry] Pruned {dropped} stale entries "
            f"({len(pruned)} live jobs remain)",
            flush=True,
        )
    return pruned


def allocate_tunnel_port(job_id: str | int, taken: set[int]) -> int:
    """Deterministic port for job_id, linear-probing past collisions.

    The base port is ``deterministic_tunnel_port(job_id)``. If another live
    worker already claims that port (extremely rare), we probe sequentially
    through the full range until we find a free slot. Probing is bounded by
    the port span, so it always terminates.

    Args:
        job_id: The SLURM job ID being allocated.
        taken: Ports already claimed by other live workers (within this
            orchestrator's view of squeue).

    Returns:
        A free port in the tunnel range.

    Raises:
        RuntimeError: If all ports in the span are taken (shouldn't happen).
    """
    base = deterministic_tunnel_port(job_id)
    for offset in range(TUNNEL_PORT_SPAN):
        candidate = TUNNEL_PORT_BASE + (
            (base - TUNNEL_PORT_BASE + offset) % TUNNEL_PORT_SPAN
        )
        if candidate not in taken:
            return candidate
    raise RuntimeError(
        f"All {TUNNEL_PORT_SPAN} tunnel ports in use — impossible under normal load"
    )


# GPU pool configuration — minimum workers per GPU type per cluster

# Concurrency divider: per-trace soft budget for complex multi-turn agentic
# traces with reflection + multiple retrievals. Kept deliberately larger than
# the input-token cap (24576) so that long traces never squeeze a second in-flight
# request off the KV cache — under this formula every GPU type ends up at
# max_concurrency=1 (weights-bound, not token-bound), which is also the honest
# operating point for 122B-A10B at TP=2+.
# max_concurrency per GPU = max(1, floor(max_model_len / TRACE_LEN_DIVIDER)).
# Average trace is ~17K tokens. Post-trace calls (reconciler + calibrator)
# are short single-turn requests that don't stack in KV cache.
# 40K divider gives 131072//40000 = 3 concurrent traces per GPU.
TRACE_LEN_DIVIDER = 40000

# All pools use max_model_len=131072 (Qwen3.5's 128K context window).
# Context budget = 95% of this (set in worker.sbatch), giving the model
# a 5% margin before the forced-answer mechanism kicks in.
GPU_POOL_CONFIG: Path = PROJECT / "src/r2c2_agent/slurm/gpu_pool.json"


def _load_gpu_pool() -> dict[tuple[str, str], dict[str, Any]]:
    """Load the per-cluster GPU pool config from `gpu_pool.json`.

    The JSON file's top-level keys are `"<cluster>/<gpu_type>"` strings;
    the orchestrator splits each on the slash into the canonical
    `(cluster, gpu_type)` tuple keys used everywhere else. `_comment` /
    `comment` fields and any other underscore-prefixed entries are
    dropped (informational only).

    Override the config path with the `GPU_POOL_CONFIG=path/to.json`
    env var. Edit `src/r2c2_agent/slurm/gpu_pool.json` to add or remove
    GPU pools — no Python edits required.
    """
    path = Path(os.environ.get("GPU_POOL_CONFIG", str(GPU_POOL_CONFIG)))
    raw = json.loads(path.read_text())
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, cfg in raw.items():
        if key.startswith("_") or not isinstance(cfg, dict):
            continue
        if "/" not in key:
            raise ValueError(f"gpu_pool key {key!r} must be '<cluster>/<gpu_type>'")
        cluster, gpu_type = key.split("/", 1)
        # Drop comment-only fields; keep everything else verbatim.
        out[(cluster, gpu_type)] = {
            k: v for k, v in cfg.items() if not k.startswith("_") and k != "comment"
        }
    return out


GPU_POOL: dict[tuple[str, str], dict[str, Any]] = _load_gpu_pool()

# Legacy flat list for --pool-config JSON compatibility
DEFAULT_WORKER_POOL: list[dict[str, Any]] = []  # unused when GPU_POOL is active

# GPU calibration — load/save calibrated max_concurrency per GPU type

CALIBRATION_FILE = PROJECT / "src/r2c2_agent/slurm/gpu_calibration.json"


def load_calibration() -> dict[str, Any]:
    """Load the GPU calibration file from disk (empty dict if missing)."""
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except Exception as exc:
            print(
                f"[calibrate] WARNING: failed to load {CALIBRATION_FILE}: {exc}",
                flush=True,
            )
    return {}


def save_calibration(cal: dict[str, Any]) -> None:
    """Persist the calibration dict to disk (atomic via tmp+rename)."""
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cal, indent=2, ensure_ascii=False))
    tmp.replace(CALIBRATION_FILE)


# Apply any existing calibrated values at import time so GPU_POOL is correct
# before any worker is created.
_calibration: dict[str, Any] = load_calibration()
_calibrated_types: set[str] = set()

for (_cal_cluster, _cal_gpu_type), _cal_cfg in GPU_POOL.items():
    _cal_key = f"{_cal_cluster}/{_cal_gpu_type}"
    if (
        _cal_key in _calibration
        and "calibrated_max_concurrency" in _calibration[_cal_key]
    ):
        _cal_old = _cal_cfg["max_concurrency"]
        _cal_cfg["max_concurrency"] = _calibration[_cal_key][
            "calibrated_max_concurrency"
        ]
        _calibrated_types.add(_cal_key)
        print(
            f"[calibrate] {_cal_key}: loaded calibrated max_concurrency="
            f"{_cal_cfg['max_concurrency']} (was {_cal_old})",
            flush=True,
        )


# Worker slot tracking


@dataclass
class WorkerSlot:
    """Tracks a GPU worker slot — its SLURM job and connection state."""

    index: int
    cluster: str
    gpu_type: str
    name: str
    gres: str
    constraint: str
    cpus: int
    mem: str
    description: str
    exclude: str = ""  # Nodes to exclude (--exclude flag)
    tunnel_port: int = 0  # For Bamboo workers
    job_id: str = ""
    state: str = "pending"  # pending, starting, ready, dead, retired
    retries: int = 0
    url: str = ""
    host: str = ""
    port: int = 0
    active: int = 0
    healthy: bool = False
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(2))
    last_health_check: float = 0.0
    last_health_ok: float = 0.0
    submit_time: float = field(default_factory=time.monotonic)
    max_model_len: int = 131072  # From pool config; used for per-slot context budget


# Job submission


async def run_cmd(cmd: str, timeout: float = 30.0) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, "", "Command timed out"
    return (
        proc.returncode or 0,
        stdout.decode().strip(),
        stderr.decode().strip(),
    )


async def probe_secondary_reachable(timeout_s: float = 10.0) -> bool:
    """10-second SSH probe to the SECONDARY_CLUSTER login node.

    Used both by ``sync_to_secondary()`` and by ``prune_unreachable_pools()``
    so code+pool reachability share one source of truth.

    Returns ``True`` iff the probe exits 0 within ``timeout_s`` seconds.
    Intentionally fast-fail: rsync's own timeout is ~120s and during a
    worst-case outage we want a single clean log line rather than
    back-to-back rsync timeouts.
    """
    probe_cmd = (
        f"ssh -o ConnectTimeout={int(timeout_s)} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=accept-new "
        f"{SECONDARY_CLUSTER} true"
    )
    probe_rc, _, _ = await run_cmd(probe_cmd, timeout=timeout_s + 5.0)
    return probe_rc == 0


def prune_unreachable_pools(keep_secondary: bool) -> list[tuple[str, str]]:
    """Remove SECONDARY_CLUSTER entries from GPU_POOL when ``keep_secondary``
    is False. Returns the list of removed (cluster, gpu_type) keys.

    Mutates the module-level GPU_POOL dict in place. Safe to call multiple
    times — no-op after the first call if nothing remains to prune.
    """
    if keep_secondary:
        return []
    removed: list[tuple[str, str]] = []
    for key in list(GPU_POOL.keys()):
        if key[0] == SECONDARY_CLUSTER:
            del GPU_POOL[key]
            removed.append(key)
    return removed


async def sync_to_secondary() -> bool:
    """Rsync code from the primary cluster to the secondary cluster.

    Runs from ``login1.{PRIMARY_CLUSTER}`` (authoritative BeeGFS view) to
    ``{SECONDARY_CLUSTER}:$PROJECT``, then md5-verifies a small list of
    critical files on both sides.

    Graceful-degrade: returns ``False`` on any failure but never raises.
    The caller is expected to log and continue — a secondary-cluster sync
    failure must not block orchestrator startup (e.g. the secondary cluster
    may be in scheduled maintenance). Set the environment variable
    ``NO_SYNC_SECONDARY=1`` to skip this function entirely — that escape
    hatch is only needed when the secondary login node itself is literally
    unreachable (full-cluster outage). During ordinary Baobab maintenance
    the compute side drains but ``login1.baobab`` stays up, so the default
    path (probe + rsync) is still correct.
    """
    # Connectivity pre-check via shared probe helper.
    if not await probe_secondary_reachable():
        print(
            f"[sync] SSH probe to {SECONDARY_CLUSTER} failed — "
            f"skipping code sync. Continuing in single-cluster mode on "
            f"{PRIMARY_CLUSTER}. Set NO_SYNC_SECONDARY=1 to silence.",
            flush=True,
        )
        return False
    print(
        f"[sync] Syncing code to {SECONDARY_CLUSTER} (via login node)...",
        flush=True,
    )
    dirs_to_sync = [
        ("src/r2c2_agent/", "src/r2c2_agent/"),
    ]
    for local_dir, remote_dir in dirs_to_sync:
        src = f"{PROJECT}/{local_dir}"
        dst = f"{SECONDARY_CLUSTER}:{PROJECT}/{remote_dir}"
        rc, _, err = await run_cmd(
            f"ssh login1.{PRIMARY_CLUSTER} 'rsync -aq --checksum"
            f' --exclude=__pycache__ --exclude="*.pyc" --exclude=logs/'
            f" {src} {dst}'",
            timeout=120.0,
        )
        if rc != 0:
            print(f"[sync] WARNING: rsync {local_dir} failed: {err}", flush=True)
            return False
    # Verify sync: compare checksums of critical files
    verify_files = [
        "src/r2c2_agent/agent/loop.py",
        "src/r2c2_agent/orchestrator/nugget_reconciler.py",
        "src/r2c2_agent/orchestrator/worker.py",
        "src/r2c2_agent/mcp_server/server.py",
    ]
    # Compare primary login node (authoritative) vs secondary for each file.
    for vf in verify_files:
        fpath = f"{PROJECT}/{vf}"
        rc, primary_md5, _ = await run_cmd(
            f"ssh login1.{PRIMARY_CLUSTER} 'md5sum {fpath}'", timeout=15.0
        )
        if rc != 0:
            print(
                f"[sync] VERIFY FAILED: cannot read {vf} on {PRIMARY_CLUSTER} login",
                flush=True,
            )
            return False
        rc, secondary_md5, _ = await run_cmd(
            f"ssh {SECONDARY_CLUSTER} 'md5sum {fpath}'", timeout=15.0
        )
        if rc != 0:
            print(
                f"[sync] VERIFY FAILED: cannot read {vf} on {SECONDARY_CLUSTER}",
                flush=True,
            )
            return False
        primary_hash = primary_md5.split()[0]
        secondary_hash = secondary_md5.split()[0]
        if primary_hash != secondary_hash:
            print(
                f"[sync] VERIFY FAILED: {vf} differs! "
                f"{PRIMARY_CLUSTER}={primary_hash[:8]} "
                f"{SECONDARY_CLUSTER}={secondary_hash[:8]}",
                flush=True,
            )
            return False
    print("[sync] Verified — all critical files match.", flush=True)
    return True


# Alias kept for callers that import sync_to_bamboo.
sync_to_bamboo = sync_to_secondary


async def submit_worker(slot: WorkerSlot, model: str, adapter: str) -> str:
    """Submit a GPU worker sbatch job. Returns job ID or empty string on failure."""
    env_vars = f"MODEL={model}"
    if adapter:
        env_vars += f" ADAPTER={adapter}"
    if slot.cluster == SECONDARY_CLUSTER and slot.tunnel_port > 0:
        env_vars += f" TUNNEL_PORT={slot.tunnel_port}"
    # PRIMARY_CLUSTER tells the worker which cluster hosts the orchestrator
    # so its tunnel target + endpoint-sync target are correct regardless
    # of which cluster the worker is on. Without this, secondary workers
    # would reverse-SSH to a hardcoded "baobab" alias.
    env_vars += f" PRIMARY_CLUSTER={PRIMARY_CLUSTER}"
    # Propagate pipeline-selector env so worker.sbatch / worker.py sees them
    # even on clusters where sbatch --export=ALL is stripped by site policy.
    for key in ("AGENTIC_PIPELINE", "POSTTRACE_PIPELINE"):
        val = os.environ.get(key)
        if val:
            env_vars += f" {key}={val}"

    # Pool-level max_model_len override. worker.sbatch normally derives
    # MAX_MODEL_LEN from per-GPU VRAM detection (a coarse 3-bucket ladder),
    # which is wrong for the 4xBW6K pool where we want a larger context
    # to unlock max_concurrency=2. Propagate the GPU_POOL value here so
    # worker.sbatch can prefer it over its fallback heuristic.
    pool_cfg = GPU_POOL.get((slot.cluster, slot.gpu_type), {})
    pool_max_model_len = pool_cfg.get("max_model_len")
    if pool_max_model_len:
        env_vars += f" MAX_MODEL_LEN={pool_max_model_len}"

    constraint_flag = ""
    if slot.constraint:
        constraint_flag = f"--constraint={slot.constraint}"

    exclude_flag = ""
    if slot.exclude:
        exclude_flag = f"--exclude={slot.exclude}"

    sbatch_args = (
        f"--parsable --job-name={slot.name} --gres={slot.gres} "
        f"--cpus-per-task={slot.cpus} --mem={slot.mem} "
        f"--time=12:00:00 {constraint_flag} {exclude_flag}"
    ).strip()

    cluster_flag = f"-M {slot.cluster}" if slot.cluster != PRIMARY_CLUSTER else ""
    cmd = f"{env_vars} sbatch {cluster_flag} {sbatch_args} {PROJECT / WORKER_SBATCH}"
    print(f"[submit-cmd] {cmd}", flush=True)

    rc, stdout, stderr = await run_cmd(cmd, timeout=30.0)
    if rc != 0:
        print(
            f"[submit] FAILED slot {slot.index} ({slot.description}): {stderr}",
            flush=True,
        )
        return ""

    # sbatch --parsable returns "JOB_ID" or "JOB_ID;cluster"
    job_id = stdout.split(";")[0].strip()
    print(
        f"[submit] Slot {slot.index} ({slot.description}) -> job {job_id}",
        flush=True,
    )
    return job_id


async def cancel_worker(slot: WorkerSlot) -> None:
    """Cancel a worker's SLURM job."""
    if not slot.job_id:
        return
    cluster_flag = f"-M {slot.cluster}" if slot.cluster != PRIMARY_CLUSTER else ""
    cmd = f"scancel {cluster_flag} {slot.job_id}"
    rc, _, err = await run_cmd(cmd, timeout=15.0)
    if rc == 0:
        print(f"[cancel] Cancelled job {slot.job_id} (slot {slot.index})", flush=True)
    else:
        print(
            f"[cancel] Failed to cancel {slot.job_id}: {err}",
            flush=True,
        )


async def check_job_state(slot: WorkerSlot) -> str:
    """Check SLURM job state. Returns: PENDING, RUNNING, COMPLETED, FAILED, etc."""
    if not slot.job_id:
        return "UNKNOWN"
    cluster_flag = f"-M {slot.cluster}" if slot.cluster != PRIMARY_CLUSTER else ""
    cmd = (
        f"squeue {cluster_flag} -j {slot.job_id} -h -o %T 2>/dev/null || "
        f"sacct {cluster_flag} -j {slot.job_id} -n -o State"
        f" --parsable2 2>/dev/null | head -1"
    )
    rc, stdout, _ = await run_cmd(cmd, timeout=10.0)
    if rc != 0 or not stdout:
        return "UNKNOWN"
    return stdout.strip().split("\n")[0].upper()


def _parse_slurm_time(s: str) -> float | None:
    """Parse a SLURM ISO-like timestamp to Unix seconds.

    SLURM emits ``%S`` (StartTime) and ``%V`` (SubmitTime) as
    ``YYYY-MM-DDTHH:MM:SS`` (local time on the cluster). ``N/A`` is
    returned for PENDING jobs' StartTime. Returns ``None`` if the string
    is empty, ``N/A``, or cannot be parsed.
    """
    if not s:
        return None
    s = s.strip()
    if not s or s.upper() in ("N/A", "UNKNOWN", "NONE"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()
    except ValueError:
        return None


async def get_job_running_seconds(slot: WorkerSlot) -> float | None:
    """Seconds the job has been in the RUNNING state, from SLURM's clock.

    Uses ``squeue -o '%T|%S|%V'`` to fetch (State, StartTime, SubmitTime).
    Returns ``now - StartTime`` when the job is RUNNING and SLURM has a
    valid StartTime. Returns ``None`` when the job is not RUNNING yet
    (PENDING / CONFIGURING / UNKNOWN) or when StartTime cannot be parsed;
    the caller should treat ``None`` as "do not fire the startup-timeout
    this cycle" and must not fall back to ``slot.submit_time``.

    The anchor matters because a job can pend in the SLURM queue for
    longer than the startup timeout. Measured from ``submit_time`` (the
    moment the orchestrator called sbatch), such a job is already past
    the guard the instant it transitions to RUNNING and would be
    cancelled seconds after starting. Measured from SLURM ``StartTime``,
    the timeout counts only the time the worker has spent starting up.
    """
    if not slot.job_id:
        return None
    cluster_flag = f"-M {slot.cluster}" if slot.cluster != PRIMARY_CLUSTER else ""
    # squeue: use %T (state), %S (StartTime), %V (SubmitTime). Pipe-
    # separated for safe splitting across clusters whose default
    # whitespace handling differs.
    cmd = f"squeue {cluster_flag} -j {slot.job_id} -h -o '%T|%S|%V' 2>/dev/null"
    rc, stdout, _ = await run_cmd(cmd, timeout=10.0)
    if rc != 0 or not stdout.strip():
        return None
    line = stdout.strip().split("\n")[0]
    parts = line.split("|")
    if len(parts) < 2:
        return None
    state = parts[0].strip().upper()
    start_time_s = parts[1].strip()
    if state != "RUNNING":
        # PENDING / CONFIGURING / COMPLETING: not "running" yet — caller
        # should skip the startup-timeout guard entirely.
        return None
    start_ts = _parse_slurm_time(start_time_s)
    if start_ts is None:
        # State says RUNNING but StartTime is N/A — rare transient
        # squeue race. Treat as "unknown age" and bail out rather than
        # using submit_time, which includes time spent in the queue.
        return None
    return max(0.0, time.time() - start_ts)


@dataclass(frozen=True)
class LiveJob:
    """A running or pending SLURM worker job seen in squeue."""

    cluster: str
    gpu_type: str
    job_id: str
    state: str  # RUNNING, PENDING, etc.
    node: str
    tunnel_port: int  # 0 for baobab, assigned port for bamboo
    host: str = ""  # for baobab direct connections (from registry)
    port: int = 0  # for baobab direct connections (from registry)


async def snapshot_live_workers() -> list[LiveJob]:
    """Return every running/pending worker job across both clusters.

    This is the *authoritative* source of which workers exist. Endpoint files
    on disk are not consulted. For Bamboo workers the deterministic tunnel
    port is computed here, with linear probing to break the (extremely rare)
    ``job_id % PORT_SPAN`` collision case.

    Returns:
        A list of LiveJob records, one per queued worker on either cluster.
    """
    taken_ports: set[int] = set()
    raw: list[tuple[str, str, str, str, str]] = []
    # (cluster, gpu_type, job_id, state, node)
    for (cluster, gpu_type), cfg in GPU_POOL.items():
        job_name = cfg["job_name"]
        cluster_flag = f"-M {cluster}" if cluster != PRIMARY_CLUSTER else ""
        cmd = (
            f"squeue {cluster_flag} --name={job_name} -u $USER "
            f"-h -o '%i %T %N' 2>/dev/null"
        )
        rc, stdout, _ = await run_cmd(cmd, timeout=15.0)
        if rc != 0 or not stdout.strip():
            continue
        for line in stdout.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            job_id = parts[0].strip("'\"")
            state = parts[1].strip("'\"").upper()
            node = parts[2].strip("'\"") if len(parts) > 2 else ""
            if state not in ("RUNNING", "PENDING", "CONFIGURING"):
                continue
            raw.append((cluster, gpu_type, job_id, state, node))

    # Port registry: load, prune stale, reuse existing assignments
    live_job_ids_all = {job_id for _, _, job_id, _, _ in raw}
    registry = load_port_registry()
    registry = prune_port_registry(registry, live_job_ids_all)
    # Pre-claim Bamboo ports from the registry so new jobs don't collide.
    for entry in registry.values():
        if entry.get("host") == "localhost":
            taken_ports.add(int(entry["port"]))

    # Sort for deterministic port assignment across restarts.
    raw.sort(key=lambda r: (r[0], int(r[2]) if r[2].isdigit() else 0))
    live: list[LiveJob] = []
    registry_changed = False
    for cluster, gpu_type, job_id, state, node in raw:
        tunnel_port = 0
        host = ""
        port = 0
        if cluster == SECONDARY_CLUSTER:
            # Cross-cluster slot: tunneled, port assigned deterministically.
            # Reuse the registry entry if present: the registry is the
            # record of assigned ports.
            if job_id in registry:
                tunnel_port = int(registry[job_id]["port"])
            else:
                tunnel_port = allocate_tunnel_port(job_id, taken_ports)
                registry[job_id] = {"host": "localhost", "port": tunnel_port}
                registry_changed = True
            taken_ports.add(tunnel_port)
            host = "localhost"
            port = tunnel_port
        else:
            # Same-cluster (PRIMARY): direct IP. Only known via registry.
            if job_id in registry:
                host = str(registry[job_id]["host"])
                port = int(registry[job_id]["port"])
        live.append(
            LiveJob(
                cluster=cluster,
                gpu_type=gpu_type,
                job_id=job_id,
                state=state,
                node=node,
                tunnel_port=tunnel_port,
                host=host,
                port=port,
            )
        )

    save_port_registry(registry)
    if registry_changed:
        print(
            f"[port-registry] Saved {len(registry)} entries to "
            f"{PORT_REGISTRY_FILE.name}",
            flush=True,
        )
    return live


def acquire_orchestrator_lock() -> None:
    """Refuse to start if another orchestrator is already running.

    Writes ``ENDPOINTS_DIR/orchestrator.lock`` with the current PID. If the
    lock exists and the PID is still alive, exits with a loud error. Stale
    locks (dead PID) are removed silently.

    Raises:
        SystemExit: If another orchestrator holds the lock.
    """
    ENDPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if ORCHESTRATOR_LOCK.exists():
        try:
            other_pid = int(ORCHESTRATOR_LOCK.read_text().strip())
        except (ValueError, OSError):
            other_pid = 0
        if other_pid > 0:
            try:
                os.kill(other_pid, 0)  # signal 0 = existence check
                print(
                    f"ERROR: orchestrator already running (PID {other_pid}, "
                    f"lock={ORCHESTRATOR_LOCK}). Refusing to start a second "
                    f"instance. If you are sure it's dead, rm the lock file.",
                    file=sys.stderr,
                    flush=True,
                )
                sys.exit(2)
            except ProcessLookupError:
                print(
                    f"[lock] Removing stale lock from dead PID {other_pid}",
                    flush=True,
                )
                ORCHESTRATOR_LOCK.unlink(missing_ok=True)
    ORCHESTRATOR_LOCK.write_text(str(os.getpid()))
    print(f"[lock] Acquired orchestrator lock (PID {os.getpid()})", flush=True)


def release_orchestrator_lock() -> None:
    """Remove the orchestrator lock file if we own it."""
    try:
        if ORCHESTRATOR_LOCK.exists():
            pid_txt = ORCHESTRATOR_LOCK.read_text().strip()
            if pid_txt == str(os.getpid()):
                ORCHESTRATOR_LOCK.unlink(missing_ok=True)
    except OSError:
        pass


async def count_pending_jobs(cluster: str, job_name: str) -> int:
    """Count PENDING SLURM jobs for a given job name on a cluster."""
    cluster_flag = f"-M {cluster}" if cluster != PRIMARY_CLUSTER else ""
    cmd = (
        f"squeue {cluster_flag} -u $USER --name={job_name} "
        f"--states=PENDING -h -o %T 2>/dev/null"
    )
    rc, stdout, _ = await run_cmd(cmd, timeout=15.0)
    if rc != 0:
        return -1  # Error — caller should skip this cycle
    if not stdout.strip():
        return 0
    return len(stdout.strip().split("\n"))


# Worker discovery + health


async def discover_endpoint(slot: WorkerSlot) -> bool:
    """Check if a worker's endpoint file exists and read it.

    If ``slot.url`` is already set (e.g. Bamboo adoption with deterministic
    port), we consider it discovered without needing the endpoint file.
    """
    if slot.url:
        return True
    ep_file = ENDPOINTS_DIR / f"worker_{slot.job_id}.endpoint"
    if not ep_file.exists():
        return False
    try:
        data = json.loads(ep_file.read_text().strip())
        slot.host = data["host"]
        slot.port = data["port"]
        slot.url = f"http://{slot.host}:{slot.port}"
        return True
    except Exception as e:
        log.warning("Failed to parse endpoint for job %s: %s", slot.job_id, e)
        return False


async def health_check(session: aiohttp.ClientSession, slot: WorkerSlot) -> bool:
    """Check if a worker is healthy via HTTP."""
    if not slot.url:
        return False
    try:
        async with session.get(
            f"{slot.url}/health",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("status") in ("ready", "busy")
    except Exception:
        pass
    return False


# Work distribution


def pick_worker(slots: list[WorkerSlot]) -> WorkerSlot | None:
    """Pick the least-loaded ready worker."""
    ready = [s for s in slots if s.state == "ready" and s.healthy]
    if not ready:
        return None
    return min(ready, key=lambda s: s.active)


def compute_breadth_first_cap(
    ready_workers: list[WorkerSlot],
    starting_workers: list[WorkerSlot],
) -> int:
    """Breadth-first dispatch cap: ``min(active) + 1`` across allocated slots.

    Rule: when N GPUs are allocated (ready or
    still loading the model), each must hold exactly the same number of
    queries (within 1) before any single slot can take another. This
    forces the dispatcher to wait for the 2nd GPU to come online before
    the 1st gets its 2nd query, etc.

    Starting slots count as virtual ``active=0`` — they'll start empty
    when they come online, so the presence of any starting slot caps
    the ready slots at active=1. Fallback (``idle_workers`` in the main
    loop) still bypasses this cap when a ready slot is truly idle
    (active=0) so a stuck-PENDING starting slot cannot starve the
    pipeline forever.

    Returns the per-slot active-count cap: a ready worker may take a
    new task iff ``worker.active < cap``.
    """
    if ready_workers:
        min_ready_active = min(w.active for w in ready_workers)
    else:
        min_ready_active = 0
    # Any starting slot contributes a virtual active=0 to the min.
    min_across_allocated = 0 if starting_workers else min_ready_active
    return min_across_allocated + 1


# Resume support


def _iter_output_path(output_path: Path, ensemble_idx: int) -> Path:
    """Resolve the per-iteration output path for ``ensemble_idx``.

    Ensemble mode writes each iteration to ``<parent>/iter_K/<name>``
    so each folder is a self-contained submission bundle.
    """
    return output_path.parent / f"iter_{ensemble_idx}" / output_path.name


def _find_record_in_file(
    path: Path,
    qid: str,
    rollout_idx: int,
) -> dict[str, Any] | None:
    """Return the first ``(qid, rollout)`` record in ``path`` or None.

    Used by the per-iter dispatch loop to detect already-completed
    iterations on resume and skip them instead of running them again.
    """
    if not path.exists():
        return None
    try:
        fh = path.open(encoding="utf-8")
    except OSError:
        return None
    with fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("question_id") != qid:
                continue
            if int(rec.get("rollout_idx", 0)) != int(rollout_idx):
                continue
            return rec
    return None


async def _wait_iter_ready(
    path: Path,
    qid: str,
    rollout_idx: int,
    *,
    timeout_s: float = 60.0,
    stability_window_s: float = 0.5,
) -> bool:
    """Poll until ``path`` contains a record for ``(qid, rollout_idx)``.

    Per-iter ensemble dispatch sends iter K to a worker only after iter
    K-1's record has been flushed to BeeGFS. The shared filesystem has
    a non-zero write-visibility window, so we also require the file
    size to be stable across ``stability_window_s`` before declaring
    the record safe to read.

    Args:
        path: ``iter_{K-1}/test_traces.jsonl`` path to wait on.
        qid: Expected ``question_id`` for the record.
        rollout_idx: Expected ``rollout_idx``.
        timeout_s: Maximum total wait in seconds. On timeout, returns
            ``False`` and the caller must decide whether to abort iter
            K or proceed with lenient prior-state loading.
        stability_window_s: After finding the record, require the file
            size to stay constant for this many seconds. Protects
            against reading a partially-flushed line.

    Returns:
        True if the record is present and the file size is stable.
        False on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            try:
                with path.open() as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("question_id") == qid and int(
                            rec.get("rollout_idx", 0)
                        ) == int(rollout_idx):
                            size1 = path.stat().st_size
                            await asyncio.sleep(stability_window_s)
                            size2 = path.stat().st_size
                            if size1 == size2:
                                return True
                            break
            except OSError:
                pass
        await asyncio.sleep(0.5)
    return False


def load_done_ids(
    output_path: Path,
    mode: str,
    ensemble_n: int = 1,
    *,
    adaptive: bool = False,
) -> set[str]:
    """Load already-completed item IDs from existing output.

    For ensemble_n > 1 (question mode): a qid/rollout pair is "done"
    only when all N iter_K/ files contain a record for it. Keys are
    returned as ``f"{qid}_{rollout_idx}"`` so they match the pending
    enumeration used downstream.

    For ``adaptive=True``: a qid is done iff the iter_K
    records satisfy the adaptive stop condition:
        - ``MAX_ADAPTIVE_ENSEMBLE`` iters are present, OR
        - at least ``MIN_ADAPTIVE_ITERATIONS`` iters are present AND the
          last two contiguous records both have ``raw_confidence <
          ADAPTIVE_LOW_CONF_THRESHOLD``.

    Placeholder records written by the retry-cap-exhausted path
    (``termination="resume_skipped"``) count as completed — the qid is
    considered terminally handled.

    Raises:
        RuntimeError: if the flat legacy path and per-iter layout both
            exist for the same run dir (operator error).
    """
    done: set[str] = set()
    if not output_path.exists() and ensemble_n <= 1 and not adaptive:
        return done

    # Ensemble-aware (question mode only, N > 1 OR adaptive).
    if mode == "question" and (ensemble_n > 1 or adaptive):
        effective_n = MAX_ADAPTIVE_ENSEMBLE if adaptive else ensemble_n
        # Coexistence guard: a flat file next to iter_K/ folders means the
        # operator switched modes mid-run, so refuse to resume.
        if output_path.exists():
            iter_k_exists = any(
                _iter_output_path(output_path, k).exists() for k in range(effective_n)
            )
            if iter_k_exists:
                raise RuntimeError(
                    f"Flat output {output_path} coexists with per-iter "
                    f"iter_*/{output_path.name} layout. Operator likely "
                    "switched --ensemble between runs against the same "
                    "output dir. Delete one of the two before resuming."
                )

        # Map (qid, rollout) → {iter_k: raw_confidence} so the adaptive
        # stop condition can be checked per qid.
        seen: dict[str, dict[int, int]] = {}
        for k in range(effective_n):
            iter_path = _iter_output_path(output_path, k)
            if not iter_path.exists():
                continue
            with open(iter_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = f"{rec['question_id']}_{rec.get('rollout_idx', 0)}"
                    except (json.JSONDecodeError, KeyError):
                        continue
                    try:
                        conf = int(rec.get("raw_confidence", 0))
                    except (TypeError, ValueError):
                        conf = 0
                    seen.setdefault(key, {})[k] = conf
        if adaptive:
            for key, iter_map in seen.items():
                k_sorted = sorted(iter_map)
                if len(k_sorted) >= MAX_ADAPTIVE_ENSEMBLE:
                    done.add(key)
                    continue
                if len(k_sorted) >= MIN_ADAPTIVE_ITERATIONS:
                    # Check consecutive pairs across the ordered iters —
                    # stop condition is any pair of adjacent low-conf
                    # records in the completed sequence.
                    for i in range(1, len(k_sorted)):
                        if (
                            iter_map[k_sorted[i]] < ADAPTIVE_LOW_CONF_THRESHOLD
                            and iter_map[k_sorted[i - 1]] < ADAPTIVE_LOW_CONF_THRESHOLD
                        ):
                            done.add(key)
                            break
            return done
        # Fixed-N: done when all N iterations wrote a record.
        done = {k for k, iters in seen.items() if len(iters) >= ensemble_n}
        return done

    # Legacy (flat) path.
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if mode == "bundle":
                    done.add(rec["bundle_id"])
                else:
                    key = f"{rec['question_id']}_{rec.get('rollout_idx', 0)}"
                    done.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# GPU calibration hook — runs once per uncalibrated GPU type


async def _run_calibration(slot: WorkerSlot, gkey_str: str) -> None:
    """Probe a worker's vLLM to find the real max_concurrency.

    Sends a POST /calibrate request to the worker's HTTP endpoint, which
    runs calibrate_gpu.calibrate() against its local vLLM internally. This works for
    both Baobab workers (direct hostname) and Bamboo workers (SSH tunnel),
    since the worker always has localhost access to its own vLLM.

    Args:
        slot: The worker slot that just became healthy.
        gkey_str: Key string like ``"baobab/a100_80"``.
    """
    global _calibration  # noqa: PLW0603 — module-level dict, updated in-place

    worker_url = slot.url.rstrip("/")
    calibrate_url = f"{worker_url}/calibrate"

    print(
        f"[calibrate] Starting calibration for {gkey_str} on slot {slot.index} "
        f"({calibrate_url}, job {slot.job_id})...",
        flush=True,
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                calibrate_url,
                json={
                    "target_tokens": 100_000,
                    "max_concurrency": 6,
                    "timeout_per_request": 180.0,
                },
                timeout=aiohttp.ClientTimeout(total=1200),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"Worker returned HTTP {resp.status}: {error_body[:200]}"
                    )
                cal_result = await resp.json()
        max_conc = cal_result.get("max_safe_concurrency", 0)
        if max_conc < 1:
            print(
                f"[calibrate] WARNING: {gkey_str} failed at concurrency=1! "
                f"Keeping max_concurrency=1",
                flush=True,
            )
            max_conc = 1

        # Cap at the pool config's max_concurrency (never exceed what we set)
        pool_key = (slot.cluster, slot.gpu_type)
        pool_cap = GPU_POOL.get(pool_key, {}).get("max_concurrency", max_conc)
        if max_conc > pool_cap:
            print(
                f"[calibrate] {gkey_str}: capping {max_conc} -> {pool_cap} "
                f"(pool config limit)",
                flush=True,
            )
            max_conc = pool_cap

        # Update in-memory pool config for all workers of this GPU type
        if pool_key in GPU_POOL:
            GPU_POOL[pool_key]["max_concurrency"] = max_conc
        _calibrated_types.add(gkey_str)

        _calibration[gkey_str] = {
            "calibrated_at_tokens": 100_000,
            "calibrated_max_concurrency": max_conc,
            "calibrated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calibration_results": cal_result.get("results", []),
        }
        save_calibration(_calibration)
        print(
            f"[calibrate] {gkey_str}: max_concurrency={max_conc} "
            f"(written to {CALIBRATION_FILE})",
            flush=True,
        )
    except Exception as exc:
        # Calibration failure should not block the worker from becoming ready.
        # Keep the default max_concurrency=1 and mark as calibrated so we
        # don't retry on every subsequent worker of the same type.
        _calibrated_types.add(gkey_str)
        print(
            f"[calibrate] ERROR calibrating {gkey_str}: {exc} — "
            "keeping max_concurrency="
            f"{GPU_POOL.get((slot.cluster, slot.gpu_type), {}).get('max_concurrency', 1)}",  # noqa: E501
            flush=True,
        )


# Main orchestration loop


async def send_question(
    session: aiohttp.ClientSession,
    slot: WorkerSlot,
    question: dict[str, Any],
    rollout_idx: int,
    temperature: float,
    context_budget: int,
    max_iterations: int,
    ensemble_n: int = 1,
) -> dict[str, Any] | None:
    """Send a question to a worker, return the rollout record or None.

    When ``ensemble_n > 1``, the worker runs the agentic loop N times
    sequentially (sharing touched-pid state) and returns
    ``{"records": [r0, r1, ..., rN-1]}``. When ``ensemble_n == 1``, the
    worker returns a single flat record dict (legacy shape).
    """
    # Use 95% of the slot's max_model_len as context budget, falling back
    # to the global --context-budget if the slot has no pool config.
    slot_budget = (
        int(slot.max_model_len * 0.95) if slot.max_model_len else context_budget
    )
    payload = {
        "question": question,
        "rollout_idx": rollout_idx,
        "temperature": temperature,
        "context_budget": slot_budget,
        "max_iterations": max_iterations,
        "ensemble": ensemble_n,
    }
    qid = question.get("question_id", "?")
    # Scale HTTP timeouts with ensemble_n so in-worker sequential iterations
    # (each ~500s on average, more on multi-hop + Phase 2 qids) don't
    # spuriously fire the client-side timeout. Baseline 20 min per iteration
    # + 10 min retry/startup headroom. The outer asyncio.timeout gives the
    # semaphore wait additional slack on top.
    per_iter_budget_s = 1200
    startup_headroom_s = 600
    http_total_s = per_iter_budget_s * max(1, ensemble_n) + startup_headroom_s
    outer_budget_s = http_total_s + 1800  # +30 min semaphore wait headroom
    try:
        async with asyncio.timeout(outer_budget_s):
            print(f"[send] {qid} -> {slot.url} (acquiring semaphore...)", flush=True)
            async with slot.semaphore:
                print(
                    f"[send] {qid} -> {slot.url} (sending HTTP, "
                    f"http_timeout={http_total_s}s, ensemble={ensemble_n})",
                    flush=True,
                )
                async with session.post(
                    f"{slot.url}/process-question",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=http_total_s),
                ) as resp:
                    if resp.status == 200:
                        print(f"[send] {qid} <- {slot.url} (200 OK)", flush=True)
                        return await resp.json()
                    else:
                        body = await resp.text()
                        print(
                            f"[send] {qid} <- {slot.url} (HTTP {resp.status})",
                            flush=True,
                        )
                        log.error(
                            "Worker %s returned %d for question %s: %s",
                            slot.url,
                            resp.status,
                            qid,
                            body,
                        )
                        return None
    except TimeoutError:
        print(
            f"[send] {qid} TIMEOUT after {outer_budget_s}s on {slot.url}",
            flush=True,
        )
        return None
    except Exception as e:
        print(f"[send] {qid} ERROR on {slot.url}: {e}", flush=True)
        log.error("Worker %s failed for question %s: %s", slot.url, qid, e)
        return None


def _resolve_prior_touched_pids(
    prior_iter_paths: list[Path],
    qid: str,
    rollout_idx: int,
) -> tuple[list[str], list[str]]:
    """Read prior iter records on the orchestrator side and union deltas.

    Eliminates a cross-cluster path-visibility race: qids 0007 and
    0033 ran with empty exclusion lists for every iteration because the
    Bamboo worker could not see the iter_K-1 file on BeeGFS even though
    it had been written by the Baobab orchestrator. The set is resolved
    here, where the file is known to be reachable, and shipped the result in
    the body as ``prior_touched_pids``. The worker prefers that
    authoritative set over re-reading the files.

    Returns:
        Tuple of (touched_pids, missing_paths) — both flat string lists.
        ``missing_paths`` is empty in normal operation; non-empty paths
        signal a partial union the caller may want to log.
    """
    from r2c2_agent.agent.ensemble import load_touched_set as _load_touched_set

    try:
        result = _load_touched_set(
            [Path(p) for p in prior_iter_paths],
            qid=qid,
            rollout_idx=rollout_idx,
            strict=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "send_question_iter: prior-state resolution failed qid=%s rollout=%d: %s",
            qid,
            rollout_idx,
            exc,
        )
        return [], [str(p) for p in prior_iter_paths]
    return sorted(result.touched), list(result.missing_paths)


async def send_question_iter(
    session: aiohttp.ClientSession,
    slot: WorkerSlot,
    question: dict[str, Any],
    rollout_idx: int,
    ensemble_idx: int,
    ensemble_n: int,
    prior_iter_paths: list[Path],
    temperature: float,
    context_budget: int,
    max_iterations: int,
    *,
    strict_prior: bool = False,
) -> dict[str, Any] | None:
    """Dispatch a single ensemble iteration K to a worker (per-iter mode).

    Unlike :func:`send_question`, which asks the worker to loop N times
    internally, this function sends ``ensemble_mode="per_iter"`` so the
    worker rebuilds ``touched_pids`` from ``prior_iter_paths`` (iter_0..
    K-1 records on BeeGFS) and runs ONE iteration.

    The orchestrator ALSO pre-resolves the prior touched-pid set from
    the same on-disk records and ships it in the body as
    ``prior_touched_pids``. The worker prefers that authoritative set
    over re-reading the files; this eliminates the cross-cluster
    BeeGFS visibility race that produced empty exclusion lists for
    qids 0007 and 0033 in an earlier run.

    Returns the single flat record dict on success, or ``None`` on HTTP
    error / timeout. A missing-prior-iter 409 response surfaces as
    ``None`` with the failure logged.
    """
    slot_budget = (
        int(slot.max_model_len * 0.95) if slot.max_model_len else context_budget
    )
    qid = question.get("question_id", "?")
    prior_touched_pids, prior_missing = _resolve_prior_touched_pids(
        prior_iter_paths, qid=str(qid), rollout_idx=rollout_idx
    )
    payload = {
        "question": question,
        "rollout_idx": rollout_idx,
        "temperature": temperature,
        "context_budget": slot_budget,
        "max_iterations": max_iterations,
        "ensemble": ensemble_n,
        "ensemble_mode": "per_iter",
        "ensemble_idx": ensemble_idx,
        "prior_iter_paths": [str(p) for p in prior_iter_paths],
        "prior_iter_strict": bool(strict_prior),
        # Orchestrator-resolved authoritative exclusion set — see
        # ``_resolve_prior_touched_pids``. Empty list when iter_0
        # (no prior) or when every prior path was missing on disk.
        "prior_touched_pids": prior_touched_pids,
        "prior_iter_missing_paths_orch": prior_missing,
    }
    per_iter_budget_s = 1200
    startup_headroom_s = 600
    http_total_s = per_iter_budget_s + startup_headroom_s
    outer_budget_s = http_total_s + 1800
    try:
        async with asyncio.timeout(outer_budget_s):
            print(
                f"[send-iter] {qid}#{ensemble_idx} -> {slot.url} "
                "(acquiring semaphore...)",
                flush=True,
            )
            async with slot.semaphore:
                print(
                    f"[send-iter] {qid}#{ensemble_idx} -> {slot.url} "
                    f"(sending HTTP, http_timeout={http_total_s}s, "
                    f"prior_paths={len(prior_iter_paths)}, "
                    f"prior_touched={len(prior_touched_pids)})",
                    flush=True,
                )
                async with session.post(
                    f"{slot.url}/process-question",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=http_total_s),
                ) as resp:
                    if resp.status == 200:
                        print(
                            f"[send-iter] {qid}#{ensemble_idx} <- {slot.url} (200 OK)",
                            flush=True,
                        )
                        return await resp.json()
                    body = await resp.text()
                    print(
                        f"[send-iter] {qid}#{ensemble_idx} <- {slot.url} "
                        f"(HTTP {resp.status})",
                        flush=True,
                    )
                    log.error(
                        "Worker %s returned %d for question %s iter %d: %s",
                        slot.url,
                        resp.status,
                        qid,
                        ensemble_idx,
                        body,
                    )
                    return None
    except TimeoutError:
        print(
            f"[send-iter] {qid}#{ensemble_idx} TIMEOUT after "
            f"{outer_budget_s}s on {slot.url}",
            flush=True,
        )
        return None
    except Exception as e:  # noqa: BLE001
        print(
            f"[send-iter] {qid}#{ensemble_idx} ERROR on {slot.url}: {e}",
            flush=True,
        )
        log.error(
            "Worker %s failed for question %s iter %d: %s",
            slot.url,
            qid,
            ensemble_idx,
            e,
        )
        return None


async def run_orchestrator(args: argparse.Namespace) -> None:
    """Main orchestration loop with dynamic GPU pool management."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    model = args.model
    adapter = args.adapter or ""

    if args.pool_config:
        pool_config = json.loads(Path(args.pool_config).read_text())
        # Legacy flat list mode — build slots directly (no pool management)
        use_gpu_pool = False
    else:
        pool_config = None
        use_gpu_pool = True

    # Dynamic SECONDARY_CLUSTER filtering. INCLUDE_SECONDARY_POOL gates this:
    #   auto (default) — probe SSH to secondary login, keep pools iff reachable.
    #   1/true/yes     — force-keep (useful to debug / when you want the
    #                    sbatch -M submission to fail cleanly rather than
    #                    silently skip).
    #   0/false/no     — force-drop without probing (fastest during a known
    #                    full-cluster outage; skips the 10s probe entirely).
    # No static single-cluster pool config is needed: GPU_POOL stays the
    # single source of truth across both clusters.
    if use_gpu_pool:
        include_secondary_raw = os.environ.get("INCLUDE_SECONDARY_POOL", "auto")
        include_secondary_flag = include_secondary_raw.strip().lower()
        if include_secondary_flag in {"1", "true", "yes", "on"}:
            keep_secondary = True
            print(
                f"[pool] INCLUDE_SECONDARY_POOL={include_secondary_raw!r} — "
                f"forcing {SECONDARY_CLUSTER} pool entries without probe",
                flush=True,
            )
        elif include_secondary_flag in {"0", "false", "no", "off"}:
            keep_secondary = False
            print(
                f"[pool] INCLUDE_SECONDARY_POOL={include_secondary_raw!r} — "
                f"skipping {SECONDARY_CLUSTER} pool entries (no probe)",
                flush=True,
            )
        else:
            keep_secondary = await probe_secondary_reachable()
            print(
                f"[pool] {SECONDARY_CLUSTER} "
                f"{'reachable' if keep_secondary else 'UNREACHABLE'} via "
                f"SSH probe — "
                f"{'keeping' if keep_secondary else 'pruning'} "
                f"{SECONDARY_CLUSTER} pool entries",
                flush=True,
            )
        removed = prune_unreachable_pools(keep_secondary=keep_secondary)
        if removed:
            print(
                f"[pool] Pruned {len(removed)} pool entr"
                f"{'y' if len(removed) == 1 else 'ies'} "
                f"({', '.join(f'{c}/{g}' for c, g in removed)}) — "
                f"running {PRIMARY_CLUSTER}-only",
                flush=True,
            )

    # Dynamic slot tracking
    slots: list[WorkerSlot] = []
    next_slot_index = 0
    # Ports currently claimed by any live/known slot. Single source of truth
    # for tunnel-port allocation. Mutated by _make_slot() and when a slot is
    # retired.
    live_tunnel_ports: set[int] = set()

    def _make_slot(
        cluster: str,
        gpu_type: str,
        cfg: dict[str, Any],
        job_id: str = "",
    ) -> WorkerSlot:
        """Create a new WorkerSlot from GPU_POOL config.

        If ``job_id`` is provided (adoption path), the tunnel port is
        computed deterministically from the job_id and probed against
        ``live_tunnel_ports``. Otherwise the port is left at 0 and assigned
        later (after sbatch returns a job_id) via
        ``_submit_with_deterministic_port``.
        """
        nonlocal next_slot_index
        gpu_label_map = {
            "a100_80": "A100-80GB",
            "h100": "H100",
            "bw6k": "BW6K",
            "h200": "H200",
        }
        # Derive GPU count from gres string: "gpu:<name>:<N>" → N
        try:
            gpu_count = int(str(cfg["gres"]).split(":")[-1])
        except (ValueError, IndexError):
            gpu_count = 2
        label = gpu_label_map.get(gpu_type, gpu_type)
        desc = f"{cluster.capitalize()} {gpu_count}x{label}"
        s = WorkerSlot(
            index=next_slot_index,
            cluster=cluster,
            gpu_type=gpu_type,
            name=cfg["job_name"],
            gres=cfg["gres"],
            constraint=cfg.get("constraint", ""),
            cpus=cfg["cpus"],
            mem=cfg["mem"],
            description=desc,
            exclude=cfg.get("exclude", ""),
            semaphore=asyncio.Semaphore(
                20
            ),  # never the bottleneck; dispatch gate throttles
            max_model_len=cfg.get("max_model_len", 131072),
        )
        next_slot_index += 1
        if cluster == SECONDARY_CLUSTER and job_id:
            port = allocate_tunnel_port(job_id, live_tunnel_ports)
            s.tunnel_port = port
            live_tunnel_ports.add(port)
        return s

    # Startup: authoritative snapshot of live workers from squeue
    print("\n--- Scanning squeue for existing worker jobs ---", flush=True)
    live_snapshot = await snapshot_live_workers()
    print(
        f"[snapshot] Found {len(live_snapshot)} live worker job(s) across clusters",
        flush=True,
    )

    # Adopt every live job with its deterministic port.
    for lj in live_snapshot:
        # GPU pool config lookup: fall back to first matching gpu_type if the
        # exact (cluster, gpu_type) key isn't present (defensive — should always match).
        cfg = GPU_POOL.get((lj.cluster, lj.gpu_type))
        if cfg is None:
            print(
                f"[adopt] WARNING: job {lj.job_id} ({lj.cluster}/{lj.gpu_type}) "
                f"has no GPU_POOL entry — skipping",
                flush=True,
            )
            continue
        s = _make_slot(lj.cluster, lj.gpu_type, cfg, job_id=lj.job_id)
        s.job_id = lj.job_id
        s.state = "starting"  # health loop will verify and upgrade to ready
        # Set url from LiveJob (populated from registry or tunnel port).
        if lj.host and lj.port > 0:
            s.host = lj.host
            s.port = lj.port
            s.url = f"http://{lj.host}:{lj.port}"
        slots.append(s)
        port_str = (
            f"tunnel={s.tunnel_port}" if lj.cluster == SECONDARY_CLUSTER else "direct"
        )
        print(
            f"[adopt] Slot {s.index} ({s.description}): "
            f"job {lj.job_id} {lj.state} on {lj.node or '?'} — {port_str}",
            flush=True,
        )

    adopted_count = len(slots)
    if adopted_count > 0:
        print(f"\nAdopted {adopted_count} existing worker(s)", flush=True)

    if use_gpu_pool and adopted_count == 0:
        # No existing workers — min_count slots are created after sbatch, in
        # the worker-submission block below, so they get real job_ids before
        # port allocation. Record what that block needs.
        initial_submissions: list[tuple[str, str, dict[str, Any]]] = []
        for (cluster, gpu_type), cfg in GPU_POOL.items():
            for _ in range(cfg["min_count"]):
                initial_submissions.append((cluster, gpu_type, cfg))
    else:
        initial_submissions = []

    if not use_gpu_pool:
        # Legacy flat-list mode: baobab-only (no tunnel) is the common case.
        for i, cfg in enumerate(pool_config or []):
            s = WorkerSlot(
                index=next_slot_index,
                cluster=cfg["cluster"],
                gpu_type=cfg.get("gpu_type", "unknown"),
                name=cfg["name"],
                gres=cfg["gres"],
                constraint=cfg.get("constraint", ""),
                cpus=cfg["cpus"],
                mem=cfg["mem"],
                description=cfg.get("description", f"Worker {i}"),
                semaphore=asyncio.Semaphore(20),  # never the bottleneck
            )
            next_slot_index += 1
            slots.append(s)
        # Legacy flat-list mode is baobab-only in practice; Bamboo tunnels
        # are not supported here because the deterministic-port path lives
        # in _submit_with_deterministic_port (GPU pool mode only).

    print("=== Self-Managing Orchestrator ===", flush=True)
    print(f"  Worker pool: {len(slots)} initial slots", flush=True)
    if use_gpu_pool:
        print("  GPU pool management: ENABLED", flush=True)
        for (cluster, gpu_type), cfg in GPU_POOL.items():
            print(
                f"    {cluster}/{gpu_type}: min_count={cfg['min_count']}",
                flush=True,
            )
    for s in slots:
        print(f"    [{s.index}] {s.description} ({s.cluster})", flush=True)
    print(flush=True)

    items: list[dict[str, Any]] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Loaded {len(items)} items from {input_path}", flush=True)

    # --regenerate-miss: purge all miss traces before loading done_ids
    if getattr(args, "regenerate_miss", False) and output_path.exists():
        kept = 0
        removed = 0
        lines_out = []
        with open(output_path) as _rf:
            for _line in _rf:
                try:
                    _rec = json.loads(_line)
                    _tm = _rec.get("trace_metadata") or {}
                    if _tm.get("answer_match"):
                        lines_out.append(_line)
                        kept += 1
                    else:
                        removed += 1
                except json.JSONDecodeError:
                    pass
        with open(output_path, "w") as _wf:
            _wf.writelines(lines_out)
        print(
            f"--regenerate-miss: kept {kept} matches, removed {removed} misses "
            f"from {output_path}",
            flush=True,
        )

    # Resume: check output file + any extra trace files
    _adaptive_mode = args.ensemble == ADAPTIVE_SENTINEL
    _effective_ensemble_for_resume = (
        MAX_ADAPTIVE_ENSEMBLE if _adaptive_mode else args.ensemble
    )
    done_ids = (
        load_done_ids(
            output_path,
            args.mode,
            ensemble_n=_effective_ensemble_for_resume,
            adaptive=_adaptive_mode,
        )
        if args.resume
        else set()
    )
    if args.done_from:
        import glob as globmod

        for pattern in args.done_from:
            for fp in sorted(globmod.glob(pattern)):
                with open(fp) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            done_ids.add(
                                f"{rec['question_id']}_{rec.get('rollout_idx', 0)}"
                            )
                        except (json.JSONDecodeError, KeyError):
                            continue
    if done_ids:
        print(f"Resume: {len(done_ids)} items already done", flush=True)

    # No separate verifier job is launched: CP2 uses a 122B self-call on
    # each worker's own vLLM instance. Workers read ``verifier.endpoint`` at
    # boot (see worker.sbatch), so a file left by another run is removed.
    verifier_endpoint_file = ENDPOINTS_DIR / "verifier.endpoint"

    if verifier_endpoint_file.exists():
        try:
            verifier_endpoint_file.unlink()
            log.info("Removed stale verifier.endpoint — CP2 uses self-call")
        except OSError:
            pass

    # VERIFIER_HOST/PORT are not propagated through the environment: that
    # would bake a stale value into every worker submitted from here. Clear
    # anything inherited from the calling shell.
    os.environ.pop("VERIFIER_HOST", None)
    os.environ.pop("VERIFIER_PORT", None)

    # Build work queue — stratified by question type (round-robin interleave)
    # This ensures all types are processed evenly, not front-loaded with one type.
    from collections import Counter, defaultdict

    pending_by_type: defaultdict[str, list[tuple[dict[str, Any], int]]] = defaultdict(
        list
    )
    for item in items:
        qid = item.get("question_id", "")
        for k in range(args.rollouts_per_item):
            key = f"{qid}_{k}"
            if key not in done_ids:
                qtype = item.get("question_type", "unknown")
                pending_by_type[qtype].append((item, k))

    # Round-robin interleave: take one from each type in rotation
    work_queue: asyncio.Queue[tuple[dict[str, Any], int]] = asyncio.Queue()
    total_work = 0
    type_keys = sorted(pending_by_type.keys())
    type_indices = {t: 0 for t in type_keys}
    exhausted = set()
    while len(exhausted) < len(type_keys):
        for t in type_keys:
            if t in exhausted:
                continue
            idx = type_indices[t]
            if idx < len(pending_by_type[t]):
                work_queue.put_nowait(pending_by_type[t][idx])
                type_indices[t] = idx + 1
                total_work += 1
            else:
                exhausted.add(t)

    # Per-type bundle totals and match/miss counters
    bundles_per_type: dict[str, int] = Counter(
        item.get("question_type", "?") for item in items
    )
    type_match: dict[str, int] = Counter()
    type_miss: dict[str, int] = Counter()
    # Seed from existing trace file
    if output_path.exists():
        with open(output_path) as _tf:
            for _line in _tf:
                if not _line.strip():
                    continue
                try:
                    _rec = json.loads(_line)
                except json.JSONDecodeError:
                    continue
                _qt = _rec.get("question_type", "?")
                _tm = _rec.get("trace_metadata") or {}
                if _tm.get("answer_match"):
                    type_match[_qt] += 1
                else:
                    type_miss[_qt] += 1

    print(f"Work queue: {total_work} items to process (stratified by type)", flush=True)
    _startup_total_m = _startup_total_mi = _startup_total_bt = 0
    for t in type_keys:
        m = type_match.get(t, 0)
        mi = type_miss.get(t, 0)
        bt = bundles_per_type.get(t, 0)
        _startup_total_m += m
        _startup_total_mi += mi
        _startup_total_bt += bt
        print(
            f"  {t:20s} queue={len(pending_by_type[t]):5d} | M{m}+X{mi}/T{bt}",
            flush=True,
        )
    print(
        f"  {'OVERALL':20s}                | "
        f"M{_startup_total_m}+X{_startup_total_mi}/T{_startup_total_bt}",
        flush=True,
    )

    if total_work == 0:
        print("Nothing to do.", flush=True)
        return

    ENDPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean stale endpoint files — only keep endpoints for jobs still in our
    # authoritative live set. Endpoint files are not authoritative for
    # tunnel ports; they only signal that a worker is ready.
    live_job_ids = {s.job_id for s in slots if s.job_id}
    for ep in ENDPOINTS_DIR.glob("worker_*.endpoint"):
        ep_job_id = ep.stem.replace("worker_", "")
        if ep_job_id in live_job_ids:
            print(f"[cleanup] Keeping live-job endpoint: {ep.name}", flush=True)
        else:
            ep.unlink()
            print(f"[cleanup] Removed stale endpoint: {ep.name}", flush=True)

    # Sync code to the secondary cluster if we have any slot over there
    # (or plan to submit one). Graceful-degrade: never abort the
    # orchestrator if sync fails — the secondary cluster may be in
    # maintenance. Operator can also set NO_SYNC_SECONDARY=1 (or
    # NO_SYNC_BAMBOO=1, its older name) to skip the sync path entirely.
    has_secondary = any(s.cluster == SECONDARY_CLUSTER for s in slots) or any(
        c == SECONDARY_CLUSTER for c, _, _ in initial_submissions
    )
    skip_secondary_sync = (
        os.environ.get("NO_SYNC_SECONDARY", "") == "1"
        or os.environ.get("NO_SYNC_BAMBOO", "") == "1"
    )
    if has_secondary and args.sync_bamboo and not skip_secondary_sync:
        try:
            ok = await sync_to_secondary()
        except Exception as exc:
            print(
                f"[sync] WARNING: sync_to_secondary raised {exc!r} — "
                f"continuing in single-cluster mode on {PRIMARY_CLUSTER}",
                flush=True,
            )
            ok = False
        if not ok:
            print(
                f"[sync] WARNING: sync to {SECONDARY_CLUSTER} failed or "
                f"verification mismatch — continuing anyway. If this is a "
                f"single-cluster run (secondary in maintenance), set "
                f"NO_SYNC_SECONDARY=1 to silence this warning.",
                flush=True,
            )
    elif skip_secondary_sync:
        print(
            f"[sync] NO_SYNC_SECONDARY=1 set — skipping sync to "
            f"{SECONDARY_CLUSTER} (single-cluster mode on {PRIMARY_CLUSTER})",
            flush=True,
        )

    async def _submit_with_deterministic_port(
        cluster: str, gpu_type: str, cfg: dict[str, Any]
    ) -> WorkerSlot | None:
        """Create a slot and submit the sbatch job, with deterministic port.

        Same-cluster (PRIMARY) jobs: submit directly, no tunnel port needed.
        Cross-cluster (SECONDARY) jobs: submit to learn the real job_id,
        compute the deterministic port, check for collision. If collision-
        free, the worker will self-compute the same port from its
        SLURM_JOB_ID — nothing more to do. If a collision is detected
        against our live set, we scancel and retry (the collision is
        recoverable because sbatch returns different job_ids each time).

        Returns the slot on success (with job_id, tunnel_port, state set),
        or None on unrecoverable failure.
        """
        slot = _make_slot(cluster, gpu_type, cfg)  # port=0 for now
        if cluster != SECONDARY_CLUSTER:
            job_id = await submit_worker(slot, model, adapter)
            if not job_id:
                slot.state = "dead"
                return None
            slot.job_id = job_id
            slot.state = "starting"
            return slot

        # Cross-cluster: up to 3 attempts to avoid deterministic-port collision.
        for attempt in range(3):
            job_id = await submit_worker(slot, model, adapter)
            if not job_id:
                slot.state = "dead"
                return None
            base = deterministic_tunnel_port(job_id)
            if base in live_tunnel_ports:
                print(
                    f"[port] Collision: job {job_id} base port {base} is "
                    f"already taken by another live slot. "
                    f"Scancelling and retrying (attempt {attempt + 1}/3).",
                    flush=True,
                )
                # Cancel the just-submitted job; we'll resubmit to get a
                # different job_id (and therefore a different port).
                slot.job_id = job_id
                await cancel_worker(slot)
                slot.job_id = ""
                continue
            # No collision — claim the port. Worker will self-compute the
            # same base port from its SLURM_JOB_ID.
            slot.job_id = job_id
            slot.tunnel_port = base
            live_tunnel_ports.add(base)
            slot.state = "starting"
            # Persist to port registry so adoption on restart finds it.
            reg = load_port_registry()
            reg[job_id] = {"host": "localhost", "port": base}
            save_port_registry(reg)
            return slot

        # All 3 attempts collided — statistically impossible, log and give up.
        print(
            f"[port] ERROR: 3 consecutive collisions for {cluster}/{gpu_type}. "
            f"live_tunnel_ports={sorted(live_tunnel_ports)}",
            flush=True,
        )
        slot.state = "dead"
        return None

    # Submit workers that don't already have a job
    if initial_submissions:
        print(
            f"\n--- Submitting {len(initial_submissions)} new workers ---",
            flush=True,
        )
        for cluster, gpu_type, cfg in initial_submissions:
            s = await _submit_with_deterministic_port(cluster, gpu_type, cfg)
            if s is not None:
                slots.append(s)
    else:
        print(
            "\n--- All workers adopted from previous run, no new submissions ---",
            flush=True,
        )

    # Main loop — monitor workers + distribute work
    completed = 0
    errors = 0
    t0 = time.monotonic()
    active_tasks: set[asyncio.Task[None]] = set()
    output_lock = asyncio.Lock()
    # Track failed items for retry
    retry_queue: asyncio.Queue[tuple[dict[str, Any], int]] = asyncio.Queue()
    # Per-item retry budget. Without it, a systemically broken qid (e.g. MCP
    # transport error on every attempt) bounces until the global
    # `errors > total_work * 3` cap trips, by which point the operator has no
    # signal short of 0/N progress. Past the budget the item is treated as
    # permanently failed.
    per_item_retry_cap: int = (
        10  # high enough to tolerate transient worker churn
    )
    retries_per_item: dict[str, int] = {}

    async def write_result(
        result: dict[str, Any],
        dest: Path | None = None,
    ) -> None:
        """Atomically append a result to the output file.

        ``dest`` routes the write to a specific path (used for ensemble
        iter_K/ folders). Defaults to the run-wide output_path.
        """
        target = dest if dest is not None else output_path
        line = json.dumps(result, ensure_ascii=False) + "\n"
        async with output_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a") as f_out:
                f_out.write(line)
                f_out.flush()
                # Push to BeeGFS so a remote worker reading iter_K before
                # iter_K+1 can see the record without waiting for the
                # shared-filesystem flush interval. Per-iter dispatch
                # depends on this.
                os.fsync(f_out.fileno())

    async def _write_ensemble_placeholders(
        item: dict[str, Any],
        rollout_idx: int,
        writer: Any,
    ) -> None:
        """Write ``resume_skipped`` placeholders for missing
        ensemble iterations after the per-item retry cap is exhausted.

        A qid whose retry budget expires mid-ensemble (say iter 0 + 1
        wrote successfully but iter 2 kept failing) stays
        permanently "partial" under ``load_done_ids`` unless we fill
        in the missing iter_K/ slot. Writing a placeholder with
        ``termination="resume_skipped"`` closes the ensemble so
        (a) ``load_done_ids`` sees all N iterations present and
        (b) downstream trace-filters (``raft_scorer.py``) can skip
        skipped records cleanly.

        Adaptive mode uses ``MAX_ADAPTIVE_ENSEMBLE`` as the effective
        iteration count via :func:`_effective_ensemble_n`. With the
        raw sentinel, ``range(args.ensemble)`` is ``range(-1)``, which
        is empty, and retry-cap-exhausted qids would be missing from
        the output stream.
        """
        await write_resume_skipped_placeholders(
            item=item,
            rollout_idx=rollout_idx,
            writer=writer,
            output_path=output_path,
            ensemble_n=_effective_ensemble_n(args.ensemble),
            temperature=args.temperature,
            adaptive=(args.ensemble == ADAPTIVE_SENTINEL),
        )

    async def process_item(
        session: aiohttp.ClientSession,
        slot: WorkerSlot,
        item: dict[str, Any],
        rollout_idx: int,
    ) -> dict[str, Any] | None:
        """Process a single item on a worker.

        Returns the raw trace result dict on success (so the caller can
        inspect trace_metadata for CP2 stats etc.) or ``None`` on failure.

        Ensemble (question mode, N>1): worker returns
        ``{"records": [r0..rN-1]}``; each record is routed to
        ``<parent>/iter_K/<name>``. We return the last record so the
        caller still gets something to log, plus the full list is
        durably on disk. If the last record has
        ``termination="ensemble_error"``, process_item returns None so
        the orchestrator's retry path re-queues the qid.
        """
        if getattr(args, "ensemble_per_iter", False) and (
            args.ensemble > 1 or args.ensemble == ADAPTIVE_SENTINEL
        ):
            qid_start = str(item.get("question_id") or item.get("qid") or "?")
            completed_last: dict[str, Any] | None = None
            ensemble_error_seen = False
            # Adaptive vs fixed dispatch. In adaptive mode we
            # roll up to MAX_ADAPTIVE_ENSEMBLE iterations, early-stopping
            # when two consecutive records have raw_confidence <
            # ADAPTIVE_LOW_CONF_THRESHOLD (after MIN_ADAPTIVE_ITERATIONS).
            adaptive_mode = args.ensemble == ADAPTIVE_SENTINEL
            max_k = MAX_ADAPTIVE_ENSEMBLE if adaptive_mode else args.ensemble
            # Two confidence lists:
            #   - recent_confs: every iter this ensemble has produced (fresh
            #     or replayed from disk on resume). Used for telemetry only
            #     (the stop-reason log prints the last-2 conf values).
            #   - fresh_confs: only iters freshly dispatched in this session.
            #     Used for the early-stop predicate. Separating them
            #     prevents ghost stops on resume when replayed records
            #     carry low confs from a prior session (or phantom zeros
            #     from parse failures on partial writes).
            recent_confs: list[int] = []
            fresh_confs: list[int] = []
            adaptive_stop_reason: str | None = None
            for k in range(max_k):
                prior_paths = [_iter_output_path(output_path, j) for j in range(k)]
                # Resume: if iter_K already has a record for (qid, rollout)
                # skip the dispatch. Prevents duplicate work when a partial
                # ensemble is re-queued after an orchestrator restart.
                dest_k = _iter_output_path(output_path, k)
                existing = _find_record_in_file(dest_k, qid_start, rollout_idx)
                if existing is not None:
                    log.info(
                        "per_iter qid=%s iter_%d already on disk; skipping",
                        qid_start,
                        k,
                    )
                    completed_last = existing
                    if adaptive_mode:
                        try:
                            recent_confs.append(int(existing.get("raw_confidence", 0)))
                        except (TypeError, ValueError):
                            # Parse fail on a corrupt partial-write record.
                            # Append to recent_confs for telemetry symmetry
                            # but do not inject a phantom zero into
                            # fresh_confs — it would ghost-trigger the stop.
                            recent_confs.append(0)
                    continue
                if k > 0:
                    prev_path = _iter_output_path(output_path, k - 1)
                    ready = await _wait_iter_ready(
                        prev_path,
                        qid=qid_start,
                        rollout_idx=rollout_idx,
                        timeout_s=120.0,
                    )
                    if not ready:
                        log.error(
                            "per_iter qid=%s: iter_%d record not visible "
                            "on BeeGFS within 120s; aborting K-chain",
                            qid_start,
                            k - 1,
                        )
                        ensemble_error_seen = True
                        break
                rec = await send_question_iter(
                    session,
                    slot,
                    item,
                    rollout_idx,
                    ensemble_idx=k,
                    # Report a fixed-equivalent ensemble_n to the worker
                    # even in adaptive mode — the record's ensemble_n field
                    # stays meaningful for downstream consumers.
                    ensemble_n=(
                        MAX_ADAPTIVE_ENSEMBLE if adaptive_mode else args.ensemble
                    ),
                    prior_iter_paths=prior_paths,
                    temperature=args.temperature,
                    context_budget=args.context_budget,
                    max_iterations=args.max_iterations,
                    strict_prior=False,
                )
                if rec is None:
                    ensemble_error_seen = True
                    break
                # Tag the record with adaptive telemetry so post-processing
                # can distinguish adaptive runs from fixed-N.
                rec["adaptive_ensemble"] = bool(adaptive_mode)
                dest = _iter_output_path(output_path, k)
                await write_result(rec, dest=dest)
                completed_last = rec
                if rec.get("termination") in (
                    "ensemble_error",
                    "missing_prior_iter",
                ):
                    ensemble_error_seen = True
                    break
                if adaptive_mode:
                    try:
                        conf_i = int(rec.get("raw_confidence", 0))
                    except (TypeError, ValueError):
                        conf_i = 0
                    recent_confs.append(conf_i)
                    fresh_confs.append(conf_i)
                    # Early stop: after at least MIN_ADAPTIVE_ITERATIONS
                    # fresh dispatches in this session, if the last two
                    # fresh-dispatched rollouts both have raw_confidence
                    # below the threshold, break. Using fresh_confs (not
                    # recent_confs) keeps replayed-from-disk iters out of
                    # the decision — they came from a prior session and
                    # may not reflect the current model state.
                    _stop, _reason = adaptive_should_stop(fresh_confs)
                    if _stop:
                        adaptive_stop_reason = _reason
                        log.info(
                            "adaptive_ensemble qid=%s stop at k=%d "
                            "(last two fresh confs=%d,%d <%d)",
                            qid_start,
                            k,
                            fresh_confs[-2],
                            fresh_confs[-1],
                            ADAPTIVE_LOW_CONF_THRESHOLD,
                        )
                        break
            if (
                adaptive_mode
                and adaptive_stop_reason is None
                and not ensemble_error_seen
            ):
                adaptive_stop_reason = "max_cap_reached"
            if adaptive_mode and completed_last is not None:
                completed_last["adaptive_stop_reason"] = adaptive_stop_reason
            if ensemble_error_seen:
                return None
            return completed_last
        else:
            result = await send_question(
                session,
                slot,
                item,
                rollout_idx,
                args.temperature,
                args.context_budget,
                args.max_iterations,
                ensemble_n=args.ensemble,
            )
        if result is None:
            return None

        # Ensemble response: {"records": [...]}. Route each record to
        # its iter_K/ folder. Retry-on-partial-fail happens at the
        # caller via `return None` when ensemble_error in last record.
        if (
            (args.ensemble > 1 or args.ensemble == ADAPTIVE_SENTINEL)
            and isinstance(result, dict)
            and "records" in result
        ):
            records = result["records"]
            last_record: dict[str, Any] | None = None
            saw_error = False
            for rec in records:
                k = int(rec.get("ensemble_idx", 0))
                dest = _iter_output_path(output_path, k)
                await write_result(rec, dest=dest)
                last_record = rec
                if rec.get("termination") == "ensemble_error":
                    saw_error = True
            if saw_error:
                # Retry path will re-queue via process_item returning None
                # (the worker's prefix-replay will fill in missing iters).
                return None
            return last_record

        # Legacy path (flat write).
        await write_result(result)
        return result

    # Pool management: cooldown with exponential backoff for persistently failing types
    # The POOL_* tunables are defined at module scope.
    pool_last_submit: dict[tuple[str, str], float] = {}  # (cluster, gpu_type) -> time
    pool_consecutive_failures: dict[tuple[str, str], int] = {}  # consecutive fail count
    pool_suspended: dict[tuple[str, str], float] = {}  # suspend timestamp per type
    pool_last_status_log: float = 0.0  # last time we logged pool status

    print("\n--- Waiting for workers to come online ---", flush=True)

    async with aiohttp.ClientSession() as session:
        worker_startup_deadline = time.monotonic() + 21600  # 6 hours

        # Cache of live job IDs per cluster with their SLURM state
        # ("RUNNING", "PENDING", "CONFIGURING", ...). Refreshed every
        # 60s. Keys on the inner dict are job_ids (the set of live jobs
        # is ``live_job_cache[cluster].keys()``); the state is used by
        # the breadth-first dispatch cap to only reserve capacity for
        # slots whose SLURM job is actually allocated (RUNNING) — not
        # those still waiting in the PENDING queue.
        live_job_cache: dict[str, dict[str, str]] = {}
        last_job_scan = 0.0

        while completed < total_work:
            now = time.monotonic()

            # Batch squeue scan every 60s to detect externally cancelled jobs
            if now - last_job_scan > 60:
                last_job_scan = now
                for cluster in ("baobab", "bamboo"):
                    flag = f"-M {cluster}" if cluster != PRIMARY_CLUSTER else ""
                    cmd = f"squeue {flag} -u $USER -h -o '%i %T' 2>/dev/null"
                    rc, stdout, _ = await run_cmd(cmd, timeout=15.0)
                    # Populate the live-job cache on any successful squeue,
                    # including the empty-result case. A cluster with zero
                    # live jobs (all cancelled externally, or never submitted
                    # on this cluster) must yield an empty cache, not a
                    # missing one: the dead-detection check at the top of the
                    # lifecycle loop short-circuits on ``live_set is None``,
                    # so adopted slots for those dead jobs would never be
                    # retired and backfill would never fire.
                    if rc == 0:
                        cache: dict[str, str] = {}
                        for line in stdout.strip().split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 2:
                                cache[parts[0]] = parts[1]
                            elif parts:
                                # Degraded: unknown state, treat as PENDING
                                # (safer — we won't reserve capacity for it).
                                cache[parts[0]] = "UNKNOWN"
                        live_job_cache[cluster] = cache

            # Monitor worker lifecycle
            ready_count = 0
            for slot in slots:
                if slot.state == "retired":
                    continue

                if slot.state == "starting":
                    # Fast check: is the job still in squeue? If not, mark dead.
                    # Grace period: skip for 90s after submission so stale
                    # cache doesn't kill a just-submitted job.
                    live_set = live_job_cache.get(slot.cluster)
                    if (
                        live_set is not None
                        and slot.job_id
                        and slot.job_id not in live_set
                        and now - slot.submit_time > 90.0
                    ):
                        print(
                            f"[worker] Slot {slot.index} job {slot.job_id} "
                            f"no longer in squeue — marking dead",
                            flush=True,
                        )
                        slot.state = "dead"
                        continue
                    if await discover_endpoint(slot):
                        ok = await health_check(session, slot)
                        if ok:
                            # GPU calibration (before marking ready)
                            gkey_str = f"{slot.cluster}/{slot.gpu_type}"
                            if gkey_str not in _calibrated_types:
                                await _run_calibration(slot, gkey_str)

                            slot.state = "ready"
                            slot.healthy = True
                            slot.last_health_ok = now
                            slot.submit_time = now  # reset grace for health checks
                            # Reset failure counter — this type works
                            gkey = (slot.cluster, slot.gpu_type)
                            if pool_consecutive_failures.get(gkey, 0) > 0:
                                print(
                                    f"[pool] {slot.cluster}/{slot.gpu_type}: "
                                    f"success — resetting failure counter",
                                    flush=True,
                                )
                                pool_consecutive_failures[gkey] = 0
                                pool_suspended.pop(gkey, None)
                            print(
                                f"[worker] Slot {slot.index} READY: "
                                f"{slot.url} ({slot.description}, job {slot.job_id})",
                                flush=True,
                            )
                    elif now - slot.submit_time > 1500:
                        # 25 min since submission only prompts a closer
                        # look. The guard itself anchors on SLURM's
                        # ``StartTime`` (``%S``), not on ``slot.submit_time``
                        # (the moment the orchestrator called sbatch): a job
                        # that pended 20 min in the queue has
                        # ``now - submit_time > 1500`` the instant it starts.
                        # See ``get_job_running_seconds``.
                        running_s = await get_job_running_seconds(slot)
                        if running_s is None:
                            # Not RUNNING yet (PENDING/CONFIGURING), or
                            # squeue raced. Fall back to a cheap state
                            # probe to decide whether to mark dead when the
                            # job fully exited the queue.
                            job_state = await check_job_state(slot)
                            if job_state in (
                                "FAILED",
                                "CANCELLED",
                                "TIMEOUT",
                                "COMPLETED",
                            ):
                                print(
                                    f"[worker] Slot {slot.index} job "
                                    f"{slot.job_id} no longer in queue "
                                    f"({job_state})",
                                    flush=True,
                                )
                                slot.state = "dead"
                            # else: PENDING / CONFIGURING / UNKNOWN — keep
                            # waiting. A job still in the queue is not killed
                            # on the submit-time anchor.
                        elif running_s > 1500:
                            # 25 min in the RUNNING state with no endpoint
                            # file — genuine startup failure.
                            print(
                                f"[worker] Slot {slot.index} job "
                                f"{slot.job_id} startup timeout "
                                f"({running_s:.0f}s RUNNING, no endpoint)",
                                flush=True,
                            )
                            slot.state = "dead"
                        # else: RUNNING but young — keep waiting

                elif slot.state == "ready":
                    # Fast check: is the job still in squeue?
                    # Grace period after becoming ready to avoid stale cache races.
                    live_set = live_job_cache.get(slot.cluster)
                    if (
                        live_set is not None
                        and slot.job_id
                        and slot.job_id not in live_set
                        and now - slot.submit_time > 90.0
                    ):
                        print(
                            f"[worker] Slot {slot.index} job {slot.job_id} "
                            f"no longer in squeue — marking dead",
                            flush=True,
                        )
                        slot.state = "dead"
                        continue
                    # Full health checks done concurrently below

            # Concurrent health checks for ready slots.
            # Run all due health checks in parallel to avoid blocking dispatch.
            hc_due = [
                s
                for s in slots
                if s.state == "ready"
                and now - s.submit_time > 300
                and now - s.last_health_check > 60
            ]
            if hc_due:
                hc_results = await asyncio.gather(
                    *(health_check(session, s) for s in hc_due),
                    return_exceptions=True,
                )
                for s, ok in zip(hc_due, hc_results):
                    s.last_health_check = now
                    if ok is True:
                        s.healthy = True
                        s.last_health_ok = now
                    else:
                        if s.healthy:
                            s.healthy = False
                            s.last_health_ok = now
                        if not s.healthy and now - s.last_health_ok > 120:
                            print(
                                f"[worker] Slot {s.index} unhealthy >2min, "
                                f"marking dead",
                                flush=True,
                            )
                            s.state = "dead"

            # Recount ready after health checks
            ready_count = sum(1 for s in slots if s.state == "ready" and s.healthy)

            # Dead slots: clean up endpoint, cancel old job, release port, retire
            for slot in slots:
                if slot.state != "dead":
                    continue
                old_ep = ENDPOINTS_DIR / f"worker_{slot.job_id}.endpoint"
                if old_ep.exists():
                    old_ep.unlink()
                await cancel_worker(slot)
                # Release tunnel port back to pool and drop from registry.
                if slot.tunnel_port > 0:
                    live_tunnel_ports.discard(slot.tunnel_port)
                    slot.tunnel_port = 0
                if slot.job_id:
                    reg = load_port_registry()
                    if slot.job_id in reg:
                        reg.pop(slot.job_id, None)
                        save_port_registry(reg)
                slot.state = "retired"
                gkey = (slot.cluster, slot.gpu_type)
                pool_consecutive_failures[gkey] = (
                    pool_consecutive_failures.get(gkey, 0) + 1
                )
                fc = pool_consecutive_failures[gkey]
                print(
                    f"[worker] Slot {slot.index} ({slot.description}) "
                    f"job {slot.job_id} retired "
                    f"({slot.cluster}/{slot.gpu_type} consecutive failures: {fc})",
                    flush=True,
                )
                if fc >= POOL_MAX_CONSECUTIVE_FAILURES:
                    pool_suspended[gkey] = now
                    print(
                        f"[pool] SUSPENDED {slot.cluster}/{slot.gpu_type}: "
                        f"{fc} consecutive failures — retry in "
                        f"{POOL_UNSUSPEND_AFTER / 60:.0f} min",
                        flush=True,
                    )

                if slot.state == "ready" and slot.healthy:
                    ready_count += 1

            # GPU pool management: pending-backfill.
            # For each (cluster, gpu_type), ensure at least 1 PENDING job
            # in the SLURM queue so backfill is instant when a worker finishes.
            if use_gpu_pool:
                has_work = not work_queue.empty() or not retry_queue.empty()
                if has_work:
                    for (cluster, gpu_type), cfg in GPU_POOL.items():
                        key = (cluster, gpu_type)

                        # Auto-unsuspend after POOL_UNSUSPEND_AFTER seconds
                        susp_time = pool_suspended.get(key)
                        if susp_time is not None:
                            if now - susp_time > POOL_UNSUSPEND_AFTER:
                                unsuspend_min = POOL_UNSUSPEND_AFTER / 60
                                print(
                                    f"[pool] UNSUSPEND {cluster}/{gpu_type}:"
                                    f" retrying after {unsuspend_min:.0f} min",
                                    flush=True,
                                )
                                pool_suspended.pop(key, None)
                                pool_consecutive_failures[key] = 0
                            else:
                                continue

                        # Exponential backoff cooldown based on consecutive failures
                        fc = pool_consecutive_failures.get(key, 0)
                        cooldown = min(
                            POOL_COOLDOWN_BASE * (2**fc),
                            POOL_COOLDOWN_MAX,
                        )
                        last = pool_last_submit.get(key, 0.0)
                        if now - last < cooldown:
                            continue

                        # Count all live slots (starting + ready) for this type.
                        # All of them are either running or will run soon, so
                        # they all count toward the target.
                        live_for_type = sum(
                            1
                            for s in slots
                            if s.cluster == cluster
                            and s.gpu_type == gpu_type
                            and s.state in ("starting", "ready")
                        )
                        # Target: min_count workers + 1 standby backfill
                        target_total = cfg["min_count"] + 1
                        if live_for_type >= target_total:
                            continue

                        print(
                            f"[pool] {cluster}/{gpu_type}: "
                            f"live={live_for_type}/{target_total} "
                            f"(failures={fc}, cooldown={cooldown:.0f}s)",
                            flush=True,
                        )
                        pool_last_submit[key] = now

                        new_slot = await _submit_with_deterministic_port(
                            cluster, gpu_type, cfg
                        )
                        if new_slot is not None:
                            slots.append(new_slot)
                            # Inject new job_id into cache so the squeue-sync
                            # check doesn't kill it before next refresh.
                            if new_slot.job_id:
                                # Assume PENDING — a fresh submission
                                # never starts in RUNNING state. The
                                # next squeue refresh will correct it.
                                live_job_cache.setdefault(new_slot.cluster, {})[
                                    new_slot.job_id
                                ] = "PENDING"
                            # Reset startup deadline — we have active workers
                            worker_startup_deadline = max(
                                worker_startup_deadline,
                                now + 1800,
                            )
                            print(
                                f"[pool] Submitted slot {new_slot.index} "
                                f"({new_slot.description}) -> job {new_slot.job_id}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[pool] Failed to submit {cluster}/{gpu_type} "
                                f"backfill",
                                flush=True,
                            )

            # Periodic pool status log
            if use_gpu_pool and now - pool_last_status_log > POOL_STATUS_INTERVAL:
                pool_last_status_log = now
                status_parts: list[str] = []
                for (c, g), cfg in GPU_POOL.items():
                    alive = sum(
                        1
                        for s in slots
                        if s.cluster == c
                        and s.gpu_type == g
                        and s.state in ("starting", "ready")
                    )
                    ready = sum(
                        1
                        for s in slots
                        if s.cluster == c
                        and s.gpu_type == g
                        and s.state == "ready"
                        and s.healthy
                    )
                    pend = await count_pending_jobs(c, cfg["job_name"])
                    fc = pool_consecutive_failures.get((c, g), 0)
                    susp = "SUSPENDED" if pool_suspended.get((c, g)) is not None else ""
                    mc = cfg.get("max_concurrency", 1)
                    ml = cfg.get("max_model_len", 0)
                    # Current active across all slots of this type
                    act = sum(
                        s.active
                        for s in slots
                        if s.cluster == c and s.gpu_type == g and s.state == "ready"
                    )
                    status_parts.append(
                        f"{c}/{g}:{ready}rdy/{alive}alive"
                        f"/pend{pend if pend >= 0 else '?'}"
                        f" ml={ml // 1000}k mc={mc} act={act}"
                        f"{f'/fail{fc}' if fc > 0 else ''}"
                        f"{f'/{susp}' if susp else ''}"
                    )
                elapsed = now - t0
                # Per-type match/miss summary
                type_summary_parts = []
                _total_m = _total_mi = _total_bt = 0
                for _ts_type in sorted(bundles_per_type):
                    _ts_m = type_match.get(_ts_type, 0)
                    _ts_mi = type_miss.get(_ts_type, 0)
                    _ts_bt = bundles_per_type[_ts_type]
                    _total_m += _ts_m
                    _total_mi += _ts_mi
                    _total_bt += _ts_bt
                    type_summary_parts.append(
                        f"{_ts_type[:6]}:M{_ts_m}+X{_ts_mi}/T{_ts_bt}"
                    )
                print(
                    f"[pool-status] {elapsed / 60:.0f}m | "
                    f"{completed}/{total_work} done | "
                    f"{' | '.join(status_parts)}",
                    flush=True,
                )
                print(
                    f"[type-status] {' | '.join(type_summary_parts)}"
                    f" -> overall:M{_total_m}+X{_total_mi}/T{_total_bt}",
                    flush=True,
                )

            # Check for startup timeout — if no workers ever come online
            if ready_count == 0:
                any_alive = any(s.state not in ("dead", "retired") for s in slots)
                if not any_alive and not use_gpu_pool:
                    # Legacy mode: no pool management to create replacements
                    print(
                        "ERROR: All worker slots dead/retired. Aborting.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                # In GPU pool mode, pool manager will keep submitting
                # replacements — only give up on hard timeout
                if now > worker_startup_deadline:
                    print(
                        "ERROR: No workers came online within 30 minutes.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                await asyncio.sleep(10)
                done_tasks = {t for t in active_tasks if t.done()}
                active_tasks -= done_tasks
                continue

            # Distribute work to available workers.
            # Move retries back to main queue before computing fair-share.
            while not retry_queue.empty():
                work_queue.put_nowait(retry_queue.get_nowait())

            # Breadth-first dispatch
            #
            # When 2 GPUs are allocated and the first becomes "ready"
            # while the second is still "starting" (loading the model),
            # a greedy fill would pile up to max_concurrency tasks on
            # the first GPU before the second comes online, leaving the
            # second idle.
            #
            # Dispatch is therefore strict round-robin
            # across all allocated slots. The cap is ``min(active) + 1``
            # where starting slots have virtual active=0. This means:
            #   - 3 GPUs allocated (1 ready A, 2 starting B/C): cap=1.
            #     A gets 1 task; wait for B and C to come online.
            #   - B ready: cap still 1 (C virtual 0). B gets 1 task.
            #   - C ready: cap still 1. C gets 1 task. Now all at 1.
            #   - cap bumps to 2. Round-robin continues until
            #     max_concurrency.
            #
            # Fallback: if a ready slot is fully idle (active=0) and
            # the cap still prevents it dispatching (because a starting
            # slot has been stuck PENDING), bypass the cap. Prevents a
            # wedged starting slot from starving the pipeline forever.
            ready_workers = [s for s in slots if s.state == "ready" and s.healthy]
            # Starting slots count toward "allocated soon-ready" seats
            # only when SLURM has actually allocated the GPU
            # (job state == "RUNNING"). A slot in state="starting" can
            # still be PENDING in the SLURM queue (could be hours), and
            # we must not reserve capacity for those — the ready workers
            # should be free to absorb all queued work when no peer is
            # actually coming online soon.
            # ``s.healthy`` is not required: it only
            # flips True when state→ready and would exclude every
            # starting slot by construction.
            starting_workers = [
                s
                for s in slots
                if s.state == "starting"
                and s.job_id
                and live_job_cache.get(s.cluster, {}).get(s.job_id) == "RUNNING"
            ]
            work_qty = work_queue.qsize()

            print(
                f"[dispatch] ready_count={ready_count}, "
                f"starting={len(starting_workers)}, "
                f"work_queue={work_qty}, "
                f"active_tasks={len(active_tasks)}",
                flush=True,
            )

            # _do_work closure — defined once, used by both Pass 1
            # (fair-share) and Pass 2 (fallback). Captures `session`,
            # `t0`, and all the counters via closure.
            async def _do_work(
                s: WorkerSlot,
                it: dict[str, Any],
                ridx: int,
            ) -> None:
                nonlocal completed, errors
                try:
                    result = await process_item(session, s, it, ridx)
                    if result is not None:
                        completed += 1
                        elapsed = time.monotonic() - t0
                        rate = completed / (elapsed / 60) if elapsed > 0 else 0
                        item_id = it.get("question_id")
                        # Per-trace summary + per-type counters
                        tm = result.get("trace_metadata", {}) or {}
                        qtype = result.get("question_type", "?")
                        matched = tm.get("answer_match", False)
                        if matched:
                            type_match[qtype] = type_match.get(qtype, 0) + 1
                        else:
                            type_miss[qtype] = type_miss.get(qtype, 0) + 1
                        conf = tm.get("confidence", 0)
                        iters = tm.get("iterations", 0)
                        tc = tm.get("tool_calls", 0)
                        term = tm.get("termination", "?")
                        wall = tm.get("wall_time_seconds", 0)
                        n_inj = int(tm.get("self_correction_injections", 0))
                        verdicts = tm.get("verifier_verdicts") or []
                        # Estimate token count from message chars
                        msgs = result.get("messages", [])
                        total_chars = sum(len(str(m.get("content", ""))) for m in msgs)
                        est_tokens = total_chars // 4  # rough estimate
                        # Quality/validity tag
                        match_tag = "MATCH" if matched else "miss"
                        saved_tag = " SAVED" if (n_inj > 0 and matched) else ""
                        cp2_tag = (
                            f" cp2={len(verdicts)}inj={n_inj}{saved_tag}"
                            if verdicts
                            else ""
                        )
                        print(
                            f"  [{completed}/{total_work}] {item_id} "
                            f"{qtype[:8]:8s} "
                            f"{match_tag:5s} "
                            f"conf={conf:3d} "
                            f"i={iters} tc={tc} "
                            f"~{est_tokens // 1000}k tok "
                            f"{wall:.0f}s "
                            f"{term} "
                            f"slot={s.index}"
                            f"{cp2_tag}"
                            f" | {rate:.1f}/min "
                            f"{elapsed / 60:.0f}m",
                            flush=True,
                        )
                    else:
                        # Per-item retry cap. Stable key from whichever id
                        # field the item carries.
                        _item_key = str(
                            it.get("bundle_id")
                            or it.get("question_id")
                            or it.get("qid")
                            or id(it)
                        )
                        retries_per_item[_item_key] = (
                            retries_per_item.get(_item_key, 0) + 1
                        )
                        if retries_per_item[_item_key] >= per_item_retry_cap:
                            print(
                                f"  [retry-cap] {_item_key} exhausted "
                                f"{per_item_retry_cap} attempts — dropped",
                                flush=True,
                            )
                            if args.ensemble > 1 or args.ensemble == ADAPTIVE_SENTINEL:
                                await _write_ensemble_placeholders(
                                    it, ridx, write_result
                                )
                            errors += 1
                        else:
                            retry_queue.put_nowait((it, ridx))
                            errors += 1
                        if errors > total_work * 3:
                            print(
                                f"ERROR: Too many errors ({errors}), aborting.",
                                file=sys.stderr,
                                flush=True,
                            )
                finally:
                    s.active -= 1  # Release capacity

            def _spawn(worker: WorkerSlot) -> None:
                """Dispatch one queued item to ``worker``: reserve
                capacity, spawn ``_do_work`` as a background task, track
                it in ``active_tasks``. Caller must have verified the
                work_queue is non-empty and the worker has spare capacity.
                """
                item, rollout_idx = work_queue.get_nowait()
                worker.active += 1
                task = asyncio.create_task(_do_work(worker, item, rollout_idx))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            # Pass 1: breadth-first dispatch. Recompute per_slot_cap on
            # each outer-while iteration so that as ready workers pick
            # up their first task the cap can bump (min_active grows by
            # 1 once every ready worker has that many tasks).
            dispatched_any = True
            while not work_queue.empty() and dispatched_any:
                dispatched_any = False
                per_slot_cap = compute_breadth_first_cap(
                    ready_workers, starting_workers
                )
                for worker in sorted(ready_workers, key=lambda s: s.active):
                    if work_queue.empty():
                        break
                    worker_max_conc = GPU_POOL.get(
                        (worker.cluster, worker.gpu_type), {}
                    ).get("max_concurrency", args.concurrency_per_worker)
                    effective_cap = min(per_slot_cap, worker_max_conc)
                    if worker.active >= effective_cap:
                        continue
                    _spawn(worker)
                    dispatched_any = True

            # Pass 2 (fallback): if the breadth-first cap held work back
            # because a starting slot appears blocked (never comes
            # online), a ready slot at active=0 should still take work
            # rather than idle. This only kicks in when starting slots
            # exist AND a ready slot is fully idle — in practice this
            # is the "starting slot stuck PENDING" escape hatch.
            if not work_queue.empty() and starting_workers:
                idle_workers = [w for w in ready_workers if w.active == 0]
                if idle_workers:
                    print(
                        f"[dispatch-fallback] queue={work_queue.qsize()}, "
                        f"idle_ready={len(idle_workers)} — bypassing "
                        f"breadth-first cap (stuck-starting escape)",
                        flush=True,
                    )
                    for worker in idle_workers:
                        if work_queue.empty():
                            break
                        _spawn(worker)

            await asyncio.sleep(2)

            done_tasks = {t for t in active_tasks if t.done()}
            active_tasks -= done_tasks

            # Safety: if errors are excessive, stop
            if errors > total_work * 3:
                break

        if active_tasks:
            print(
                f"\nWaiting for {len(active_tasks)} in-flight tasks...",
                flush=True,
            )
            await asyncio.gather(*active_tasks, return_exceptions=True)

    # Cancel all workers
    print("\n--- Cancelling workers ---", flush=True)
    cancel_tasks = [
        cancel_worker(s) for s in slots if s.job_id and s.state in ("starting", "ready")
    ]
    await asyncio.gather(*cancel_tasks, return_exceptions=True)

    for ep in ENDPOINTS_DIR.glob("worker_*.endpoint"):
        ep.unlink(missing_ok=True)

    # Summary
    total_time = time.monotonic() - t0
    print(f"\n{'=' * 60}", flush=True)
    print("=== Orchestrator Complete ===", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Completed: {completed}/{total_work}", flush=True)
    print(f"  Errors:    {errors}", flush=True)
    print(f"  Time:      {total_time:.0f}s ({total_time / 3600:.1f}h)", flush=True)
    if completed > 0:
        print(f"  Avg/item:  {total_time / completed:.1f}s", flush=True)
    print(f"  Output:    {output_path}", flush=True)

    # Worker slot summary
    print(f"\n  Worker slots ({len(slots)} total created):", flush=True)
    for s in slots:
        print(
            f"    [{s.index}] {s.description} ({s.cluster}/{s.gpu_type}): "
            f"state={s.state}, job={s.job_id}",
            flush=True,
        )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Self-managing distributed orchestrator"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL (bundles or questions)"
    )
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument(
        "--mode",
        choices=["question"],
        default="question",
        help="``question`` runs the agentic loop on input questions JSONL.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rollouts-per-item", type=int, default=1)
    parser.add_argument(
        "--ensemble",
        type=int,
        nargs="?",
        const=-1,  # sentinel: --ensemble without a value means adaptive
        default=1,
        help=(
            "Run each question N times sequentially with passages assessed "
            "by prior iterations subtracted from the retrieval index. "
            "When N>1, output is written to "
            "``<output-dir>/iter_K/<output-filename>`` per iteration. "
            "Default 1 = legacy behaviour (flat output path). "
            "Pass the flag WITHOUT a value (--ensemble) to enable adaptive "
            "mode: start at 0, stop when two consecutive rollouts have "
            "raw_confidence < 50 (after min 3 iters), hard cap at 10."
        ),
    )
    parser.add_argument(
        "--ensemble-per-iter",
        action="store_true",
        help=(
            "Node-independent ensemble dispatch. Instead of asking one "
            "worker to loop N times with in-memory touched_pids state, "
            "the orchestrator issues N HTTP calls (one per ensemble slot) "
            "and each worker reads prior iter_K records from BeeGFS to "
            "reconstruct the exclusion set. Makes resume from partial "
            "ensembles correct and allows cross-node dispatch of the "
            "K-chain. Requires --ensemble > 1."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--context-budget", type=int, default=124518)  # 95% of 131072
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--regenerate-miss",
        action="store_true",
        help=(
            "Before starting, remove all answer_match=False traces from "
            "the output file. This requeues missed bundles for retry with "
            "the current (potentially fixed) code."
        ),
    )
    parser.add_argument(
        "--done-from",
        nargs="*",
        default=[],
        help="Glob patterns for existing trace files to skip",
    )
    parser.add_argument("--concurrency-per-worker", type=int, default=2)
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="HF model ID for workers",
    )
    parser.add_argument("--adapter", default="", help="LoRA adapter path")
    parser.add_argument(
        "--pool-config",
        default="",
        help="Path to JSON file defining the worker pool (default: built-in)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max resubmits per worker slot",
    )
    parser.add_argument(
        "--sync-bamboo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sync code to Bamboo before starting (default: true)",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        default=False,
        help=(
            "Rsync code to the secondary cluster and exit (no orchestrator "
            "launch, no lock). Use to push changes between runs without "
            "starting a full orchestrator. Honours NO_SYNC_SECONDARY "
            "(will still probe and exit cleanly)."
        ),
    )
    args = parser.parse_args()

    if args.ensemble_per_iter and args.ensemble == 1:
        parser.error(
            "--ensemble-per-iter requires --ensemble > 1 or bare --ensemble (adaptive)"
        )
    if args.ensemble == 0 or (args.ensemble < 0 and args.ensemble != ADAPTIVE_SENTINEL):
        parser.error("--ensemble must be a positive integer or the bare flag")
    # Adaptive ensemble (bare --ensemble) requires --ensemble-per-iter. The
    # in-worker "legacy" sequential path has no adaptive stop predicate and
    # the worker clamps ensemble_n = max(1, -1) = 1, which silently runs a
    # single rollout and is not what the caller asked for, so refuse.
    if args.ensemble == ADAPTIVE_SENTINEL and not args.ensemble_per_iter:
        parser.error(
            "bare --ensemble (adaptive) requires --ensemble-per-iter — the "
            "in-worker sequential path has no adaptive stop predicate"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.sync_only:
        # Standalone sync — no orchestrator lock, no SLURM submission.
        ok = asyncio.run(sync_to_secondary())
        raise SystemExit(0 if ok else 2)

    acquire_orchestrator_lock()
    try:
        asyncio.run(run_orchestrator(args))
    finally:
        release_orchestrator_lock()


if __name__ == "__main__":
    main()

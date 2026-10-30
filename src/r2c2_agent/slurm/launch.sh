#!/bin/bash
# Launch the self-managing orchestrator.
# It submits GPU workers (sbatch), monitors them, distributes work, and
# cancels when done. This is the script that drives the actual agent run
# across multiple GPU machines — it produces test_traces.jsonl, then the
# ensemble aggregator emits BITEM-PR.zip + BITEM-AC.zip from those traces.
#
# Usage:
#   bash src/r2c2_agent/slurm/launch.sh                  # production (env defaults)
#   bash src/r2c2_agent/slurm/launch.sh --demo           # smoke test (small model + K=2)
#   OUTPUT=/path/to.jsonl bash src/r2c2_agent/slurm/launch.sh
#
# Multi-cluster failover
# ----------------------
# This script is cluster-aware. It detects whether it is running on Baobab
# or Bamboo via the hostname and advertises the matching cross-cluster
# `tail -F` helper. The sbatch submission goes to the local cluster's
# scheduler; orchestrator.py picks up PRIMARY_CLUSTER from its environment.
#
# The orchestrator runs a 10s SSH probe to the secondary cluster login at
# startup. If it fails, secondary pool entries are pruned from GPU_POOL
# automatically and the run proceeds single-cluster. GPU_POOL stays the
# single source of truth — no pool JSON file needed.
#
# Override the auto-probe with INCLUDE_SECONDARY_POOL:
#   INCLUDE_SECONDARY_POOL=auto (default) — probe + decide
#   INCLUDE_SECONDARY_POOL=0              — force single-cluster, skip probe
#   INCLUDE_SECONDARY_POOL=1              — force include
#
# Common overrides (all passed through to orchestrator.sbatch):
#   INPUT=...         — input questions JSONL (default: data/r2c2/test_questions.jsonl)
#   OUTPUT=...        — output JSONL path (default: $PROJECT/data/runs/<run>/test_traces.jsonl)
#   MODEL=...         — HF model ID (default: Qwen/Qwen3.5-122B-A10B-FP8)
#   ENSEMBLE=...      — N (fixed) or "adaptive" (default: 1)
#   POOL_CONFIG=...   — path to worker pool JSON (forces single-cluster)
#
# --demo (smoke test) defaults
# ----------------------------
# When --demo is passed, the launcher sets the following env vars *only
# if the caller did not already export them*:
#   INPUT      = $PROJECT/data/r2c2/dry_run_questions.jsonl
#   OUTPUT     = $PROJECT/submission/demo-<ts>/test_traces.jsonl
#                (parallel to submission/pr-2026-04-25 + ac-2026-05-15
#                 — `submission/demo-*` is gitignored)
#   MODE       = question
#   ENSEMBLE   = 2                       # K=2 rollouts instead of adaptive K=4
#   ENSEMBLE_PER_ITER = 1
#   PR_RUN_NAME       = BITEM-PG-2
#
# The MODEL, MAX_ITERATIONS, GPU count, and partition are **not** changed
# — demo and production use the same Qwen3.5-122B-A10B-FP8 teacher on
# the same GPU pool. Demo speed comes from fewer questions and fewer
# rollouts, not a smaller model.

set -euo pipefail

DEMO=0
for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
        -h|--help) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "WARN: unknown arg: $arg" >&2 ;;
    esac
done

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

if [ "$DEMO" = 1 ]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    export INPUT="${INPUT:-$PROJECT/data/r2c2/dry_run_questions.jsonl}"
    export OUTPUT="${OUTPUT:-$PROJECT/submission/demo-${TS}/test_traces.jsonl}"
    export MODE="${MODE:-question}"
    export ENSEMBLE="${ENSEMBLE:-2}"
    export ENSEMBLE_PER_ITER="${ENSEMBLE_PER_ITER:-1}"
    export PR_RUN_NAME="${PR_RUN_NAME:-BITEM-PG-2}"
    echo "launch.sh: --demo defaults applied (same MODEL + GPU pool as production)"
    echo "  INPUT=$INPUT"
    echo "  OUTPUT=$OUTPUT"
    echo "  ENSEMBLE=$ENSEMBLE"
fi

# Cluster detection (matches orchestrator.py _detect_primary_cluster)
HOST_LOWER="$(hostname -f 2>/dev/null | tr '[:upper:]' '[:lower:]')"
[ -z "$HOST_LOWER" ] && HOST_LOWER="$(hostname | tr '[:upper:]' '[:lower:]')"
CLUSTER="${PRIMARY_CLUSTER:-}"
if [ -z "$CLUSTER" ]; then
    case "$HOST_LOWER" in
        *bamboo*) CLUSTER="bamboo" ;;
        *baobab*) CLUSTER="baobab" ;;
        *)        CLUSTER="baobab" ;;
    esac
fi
OTHER_CLUSTER="bamboo"
[ "$CLUSTER" = "bamboo" ] && OTHER_CLUSTER="baobab"

ENDPOINTS_DIR="$PROJECT/src/r2c2_agent/slurm/endpoints"
LOGS_DIR="$PROJECT/src/r2c2_agent/slurm/logs"
mkdir -p "$ENDPOINTS_DIR" "$LOGS_DIR"

echo "Submitting self-managing orchestrator on cluster: $CLUSTER (other: $OTHER_CLUSTER)"
JOB_ID=$(sbatch --parsable \
    --time=12:00:00 \
    --export=ALL \
    "$PROJECT/src/r2c2_agent/slurm/orchestrator.sbatch")

echo "Orchestrator job: $JOB_ID"
echo ""
echo "Monitor:"
echo "  tail -f $LOGS_DIR/orchestrator_${JOB_ID}.out"
echo ""
echo "Cancel everything (orchestrator will cancel its workers on exit):"
echo "  scancel $JOB_ID"

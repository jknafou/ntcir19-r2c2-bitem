#!/bin/bash
# CPU job that runs the self-managing orchestrator.
# Submits GPU workers, monitors lifecycle, distributes work, cancels on exit.
#
# Environment variables (set by launch.sh):
#   INPUT             — input questions JSONL (default: data/r2c2/test_questions.jsonl)
#   OUTPUT            — output JSONL path
#   MODE              — must be "question" (only mode supported)
#   TEMPERATURE       — sampling temperature (default: 0.0)
#   ENSEMBLE          — N (fixed) or "adaptive" (default: 1)
#   ENSEMBLE_PER_ITER — 1 enables per-iter ensemble dispatch
#   ROLLOUTS          — rollouts per item (default: 1)
#   MAX_ITERATIONS    — max agentic loop iterations (default: 50)
#   CONCURRENCY       — concurrent requests per worker (default: 2)
#   MODEL             — HF model ID (default: Qwen/Qwen3.5-122B-A10B-FP8)
#   ADAPTER           — LoRA adapter path (optional)
#   POOL_CONFIG       — path to worker pool JSON (optional)
#   PR_RUN_NAME       — run name embedded in PR/AC files
#   GENERATE_SUBMISSION — 1 (default) emits PR+AC after the run
#
# Submit via launch.sh; do not invoke directly.

#SBATCH --job-name=orch-ctrl
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_agent/slurm/logs/orchestrator_%j.out
#SBATCH --error=src/r2c2_agent/slurm/logs/orchestrator_%j.err

set -euo pipefail

PROJECT="${PROJECT:-$(pwd)}"

INPUT="${INPUT:-$PROJECT/data/r2c2/test_questions.jsonl}"
OUTPUT="${OUTPUT:-$PROJECT/data/runs/run_$SLURM_JOB_ID/test_traces.jsonl}"
DONE_FROM="${DONE_FROM:-}"
MODE="${MODE:-question}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ROLLOUTS="${ROLLOUTS:-1}"
ENSEMBLE="${ENSEMBLE:-1}"
ENSEMBLE_PER_ITER="${ENSEMBLE_PER_ITER:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
CONCURRENCY="${CONCURRENCY:-2}"
MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
ADAPTER="${ADAPTER:-}"
POOL_CONFIG="${POOL_CONFIG:-}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NO_SYNC_SECONDARY="${NO_SYNC_SECONDARY:-0}"
GENERATE_SUBMISSION="${GENERATE_SUBMISSION:-1}"
PR_RUN_NAME="${PR_RUN_NAME:-BITEM-PG-1}"
LLM_URL="${LLM_URL:-}"

ENDPOINTS_DIR="$PROJECT/src/r2c2_agent/slurm/endpoints"
LOGS_DIR="$PROJECT/src/r2c2_agent/slurm/logs"
mkdir -p "$ENDPOINTS_DIR" "$LOGS_DIR"

echo "=== Self-Managing Orchestrator ==="
echo "  Job:          $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  Input:        $INPUT"
echo "  Output:       $OUTPUT"
echo "  Mode:         $MODE"
echo "  Ensemble:     $ENSEMBLE (per-iter=$ENSEMBLE_PER_ITER)"
echo "  Model:        $MODEL"
echo "  Adapter:      ${ADAPTER:-none}"
echo "  Start:        $(date -Iseconds)"

# Clean stale submission files from the output directory
OUTPUT_DIR="$(dirname "$OUTPUT")"
if [ -d "$OUTPUT_DIR" ]; then
    rm -f "$OUTPUT_DIR"/BITEM-*.PR.txt "$OUTPUT_DIR"/BITEM-*.AC.txt 2>/dev/null
    if [ "$ENSEMBLE" = "adaptive" ]; then
        _CLEAN_N=10
    elif [ "$ENSEMBLE" -gt 1 ] 2>/dev/null; then
        _CLEAN_N="$ENSEMBLE"
    else
        _CLEAN_N=0
    fi
    if [ "$_CLEAN_N" -gt 1 ]; then
        for k in $(seq 0 $((_CLEAN_N - 1))); do
            rm -f "$OUTPUT_DIR/iter_$k"/BITEM-*.PR.txt \
                  "$OUTPUT_DIR/iter_$k"/BITEM-*.AC.txt 2>/dev/null
        done
    fi
fi

# Pre-create SSH local forwards for cross-cluster reverse tunnels
_HOST_LOWER="$(hostname -f 2>/dev/null | tr '[:upper:]' '[:lower:]')"
[ -z "$_HOST_LOWER" ] && _HOST_LOWER="$(hostname | tr '[:upper:]' '[:lower:]')"
PRIMARY_CLUSTER="${PRIMARY_CLUSTER:-}"
if [ -z "$PRIMARY_CLUSTER" ]; then
    case "$_HOST_LOWER" in
        *bamboo*) PRIMARY_CLUSTER="bamboo" ;;
        *baobab*) PRIMARY_CLUSTER="baobab" ;;
        *)        PRIMARY_CLUSTER="baobab" ;;
    esac
fi
export PRIMARY_CLUSTER

TUNNEL_PIDS=""
echo "Setting up local forwards to login1.${PRIMARY_CLUSTER} for tunnel ports 19000-19999..."
SSH_FORWARDS=""
for PORT in $(seq 19000 19999); do
    SSH_FORWARDS="$SSH_FORWARDS -L ${PORT}:localhost:${PORT}"
done
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    $SSH_FORWARDS -N "login1.${PRIMARY_CLUSTER}" &
TUNNEL_PIDS="$!"

cleanup_tunnels() {
    for PID in $TUNNEL_PIDS; do
        kill "$PID" 2>/dev/null || true
    done
}
trap cleanup_tunnels EXIT

# Build CLI args
SYNC_FLAG="--sync-bamboo"
[ "$NO_SYNC_SECONDARY" = "1" ] && SYNC_FLAG="--no-sync-bamboo"

POOL_FLAG=""
[ -n "$POOL_CONFIG" ] && POOL_FLAG="--pool-config $POOL_CONFIG"

ADAPTER_FLAG=""
[ -n "$ADAPTER" ] && ADAPTER_FLAG="--adapter $ADAPTER"

PER_ITER_FLAG=""
[ "$ENSEMBLE_PER_ITER" = "1" ] && PER_ITER_FLAG="--ensemble-per-iter"

if [ "$ENSEMBLE" = "adaptive" ]; then
    ENSEMBLE_FLAG="--ensemble"
else
    ENSEMBLE_FLAG="--ensemble $ENSEMBLE"
fi

DONE_FROM_FLAG=""
[ -n "$DONE_FROM" ] && DONE_FROM_FLAG="--done-from $DONE_FROM"

# Run the self-managing orchestrator
PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
"$PROJECT/.venv/bin/python3" -m r2c2_agent.orchestrator.orchestrator \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --mode "$MODE" \
    --temperature "$TEMPERATURE" \
    --rollouts-per-item "$ROLLOUTS" \
    $ENSEMBLE_FLAG \
    $PER_ITER_FLAG \
    --max-iterations "$MAX_ITERATIONS" \
    --concurrency-per-worker "$CONCURRENCY" \
    --model "$MODEL" \
    $ADAPTER_FLAG \
    $POOL_FLAG \
    --max-retries "$MAX_RETRIES" \
    $SYNC_FLAG \
    --resume \
    $DONE_FROM_FLAG

echo ""
echo "=== Orchestrator job $SLURM_JOB_ID complete ==="

# Generate submission via the ensemble aggregator
if [ "$GENERATE_SUBMISSION" = "1" ]; then
    echo ""
    echo "=== Generating Submission via ensemble_aggregator ==="
    RUN_DIR="$(dirname "$OUTPUT")"
    PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 \
    "$PROJECT/.venv/bin/python3" -m r2c2_agent.orchestrator.ensemble_aggregator \
        --run-dir "$RUN_DIR" \
        --run-name "$PR_RUN_NAME" \
        --source-questions "$INPUT" \
        --strict
fi

echo "End: $(date -Iseconds)"

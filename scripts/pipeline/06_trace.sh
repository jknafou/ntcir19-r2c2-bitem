#!/usr/bin/env bash
# Phase 6 — Agent trace generation (single command).
#
# Thin wrapper around src/r2c2_agent/slurm/launch.sh — submits the
# orchestrator (CPU controller) which itself sbatch's a pool of GPU
# workers across Baobab + Bamboo per src/r2c2_agent/slurm/gpu_pool.json.
# Each worker bundles vLLM + MCP + ES + NLI in one job. The
# orchestrator dispatches per-question work to the pool; when all 65
# qids are processed, the ensemble aggregator emits BITEM-PR.zip +
# BITEM-AC.zip.
#
# Production:
#   - INPUT  = data/r2c2/test_questions.jsonl (65 qids)
#   - MODEL  = Qwen/Qwen3.5-122B-A10B-FP8
#   - ENSEMBLE = adaptive (up to K=4)
#   - OUTPUT = data/runs/pr_<ts>/test_traces.jsonl
#
# Demo (same MODEL, same GPU pool):
#   - INPUT  = data/r2c2/dry_run_questions.jsonl (5 qids)
#   - ENSEMBLE = 2 (fixed K=2 — no adaptive)
#   - OUTPUT = submission/demo-<ts>/test_traces.jsonl
#
# Usage:
#   bash scripts/pipeline/06_trace.sh
#   bash scripts/pipeline/06_trace.sh --demo

set -euo pipefail

DEMO=0
for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
        -h|--help) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [ "$DEMO" = 1 ]; then
    bash src/r2c2_agent/slurm/launch.sh --demo
else
    bash src/r2c2_agent/slurm/launch.sh
fi

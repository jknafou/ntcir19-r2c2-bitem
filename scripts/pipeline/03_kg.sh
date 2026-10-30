#!/usr/bin/env bash
# Phase 3 — Knowledge-graph build (single command).
#
# Submits:
#   1. build_kg.sbatch         (16-way array, map phase per enriched shard)
#   2. build_kg_reduce.sbatch  (depends on map; merges + computes NPMI)
#
# Reads $CHUNKS_DIR/passages_enriched_v3.jsonl (output of Phase 2 NER).
# Writes $INDEXES_DIR/bitem-kg.json (the KG consumed by the agent's
# MCP server at runtime).
#
# Usage:
#   bash scripts/pipeline/03_kg.sh
#   bash scripts/pipeline/03_kg.sh --demo

set -euo pipefail

DEMO=0
for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
        -h|--help) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    esac
done
DEMO_ARG=""
[ "$DEMO" = 1 ] && DEMO_ARG="--demo"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Phase 3: knowledge-graph build $([ "$DEMO" = 1 ] && echo "(--demo)") ==="

MAP_JID=$(sbatch --parsable src/r2c2_index/slurm/build_kg.sbatch $DEMO_ARG)
echo "  [1/2] kg-map     job=$MAP_JID (16-way array)"

REDUCE_JID=$(sbatch --parsable \
    --dependency=afterok:$MAP_JID \
    src/r2c2_index/slurm/build_kg_reduce.sbatch $DEMO_ARG)
echo "  [2/2] kg-reduce  job=$REDUCE_JID (afterok:$MAP_JID)"

echo ""
echo "Final artefact (when reduce finishes):"
if [ "$DEMO" = 1 ]; then
    echo "  $REPO_ROOT/data/r2c2_corpus_demo_N10000/indexes/bitem-kg.json"
else
    echo "  $REPO_ROOT/data/indexes/bitem-kg.json"
fi

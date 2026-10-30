#!/usr/bin/env bash
# Phase 5 — FAISS dense indexes (single command).
#
# Submits in parallel:
#   1. build_faiss_plain.sbatch     bitem-faiss-plain    (passages.jsonl)
#   2. build_faiss_enriched.sbatch  bitem-faiss-enriched (passages_enriched_v3.jsonl)
#
# Each is a single GPU job — they don't depend on each other and run
# in parallel. Both encode with a sentence-transformer and build a
# FAISS IndexFlatIP.
#
# Usage:
#   bash scripts/pipeline/05_faiss.sh
#   bash scripts/pipeline/05_faiss.sh --demo

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

echo "=== Phase 5: FAISS dense indexes $([ "$DEMO" = 1 ] && echo "(--demo)") ==="

PLAIN_JID=$(sbatch --parsable src/r2c2_index/slurm/build_faiss_plain.sbatch $DEMO_ARG)
ENRICH_JID=$(sbatch --parsable src/r2c2_index/slurm/build_faiss_enriched.sbatch $DEMO_ARG)
echo "  [1/2] faiss-plain     job=$PLAIN_JID"
echo "  [2/2] faiss-enriched  job=$ENRICH_JID (parallel)"

echo ""
echo "Final artefacts (when both jobs finish):"
if [ "$DEMO" = 1 ]; then
    echo "  $REPO_ROOT/data/r2c2_corpus_demo_N10000/indexes/bitem-faiss-{plain,enriched}/"
else
    echo "  $REPO_ROOT/data/indexes/bitem-faiss-{plain,enriched}/"
fi

#!/usr/bin/env bash
# Phase 4 — Elasticsearch indexes (single command).
#
# Submits:
#   1. start_es.sbatch           long-running ES service job (writes
#                                src/r2c2_index/slurm/logs/.es_url when ready)
#   2. (wait inside) until $LOGS/.es_url exists
#   3. index_es_base.sbatch      16-way array — bitem-passages         (~2M docs)
#   4. index_es_ner.sbatch       16-way array — bitem-passages-ner     (~2M docs)
#   5. index_es_docs.sbatch      4-way  array — bitem-documents
#   6. finalize_es.sbatch        depends on all three index_es_*; force-merge
#
# The ES service is left running so the agent can hit it during Phase 6
# (it has --time=12h on start_es.sbatch). The agent's worker.sbatch
# spawns its own ES; this Phase 4 ES is only used during indexation.
# scancel it manually once Phase 4 finishes if you don't need it.
#
# Usage:
#   bash scripts/pipeline/04_es.sh
#   bash scripts/pipeline/04_es.sh --demo

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

LOGS=src/r2c2_index/slurm/logs

echo "=== Phase 4: Elasticsearch indexes $([ "$DEMO" = 1 ] && echo "(--demo)") ==="

# Drop any previous .es_url so we wait on the new one.
rm -f "$LOGS/.es_url"

# 1) ES service (background — completes only at --time=12h or scancel)
ES_JID=$(sbatch --parsable src/r2c2_index/slurm/start_es.sbatch $DEMO_ARG)
echo "  [1/4] start_es        job=$ES_JID (long-running)"

# 2) Wait for the ES URL file (the service writes it ~3-5 min after the
#    sbatch is allocated). All downstream indexer arrays depend on
#    afterstart:$ES_JID OR on the file appearing — we use a small wait
#    job to gate on the URL file, then chain index_es_* on afterok:$wait.
WAIT_WRAP=$(cat <<EOF
until [ -f "${REPO_ROOT}/${LOGS}/.es_url" ]; do sleep 15; done
echo "ES ready: \$(cat ${REPO_ROOT}/${LOGS}/.es_url)"
EOF
)
WAIT_JID=$(sbatch --parsable \
    --job-name=es-wait --partition=shared-cpu --cpus-per-task=1 \
    --mem=1G --time=01:00:00 \
    --output="$LOGS/es_wait_%j.out" \
    --wrap="$WAIT_WRAP")
echo "  [2/4] es-wait         job=$WAIT_JID (polls $LOGS/.es_url)"

# 3) Three parallel index arrays — all depend on the wait completing
BASE_JID=$(sbatch --parsable --dependency=afterok:$WAIT_JID \
    src/r2c2_index/slurm/index_es_base.sbatch $DEMO_ARG)
NER_JID=$(sbatch --parsable --dependency=afterok:$WAIT_JID \
    src/r2c2_index/slurm/index_es_ner.sbatch $DEMO_ARG)
DOCS_JID=$(sbatch --parsable --dependency=afterok:$WAIT_JID \
    src/r2c2_index/slurm/index_es_docs.sbatch $DEMO_ARG)
echo "  [3/4] index arrays    base=$BASE_JID ner=$NER_JID docs=$DOCS_JID"

# 4) finalize after all three indexers complete
FINAL_JID=$(sbatch --parsable \
    --dependency=afterok:$BASE_JID:$NER_JID:$DOCS_JID \
    src/r2c2_index/slurm/finalize_es.sbatch $DEMO_ARG)
echo "  [4/4] finalize_es     job=$FINAL_JID (afterok:$BASE_JID:$NER_JID:$DOCS_JID)"

echo ""
echo "Phase 4 submitted. Once finalize_es completes, scancel the ES service:"
echo "  scancel $ES_JID"

#!/bin/bash
#SBATCH --job-name=r2c2-es-base
#SBATCH --partition=shared-cpu
## #SBATCH --nodelist=cpu330  # uncomment + edit to pin ES traffic to a single CPU node
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/index_es_base_%A_%a.out
#SBATCH --error=src/r2c2_index/slurm/logs/index_es_base_%A_%a.err
#
# Resources:
#   shared-cpu: smallest node has 12 CPUs and 62G RAM; 12h limit.
#   16 parallel tasks, each bulk-indexing ~130K passages into the same ES index.
#   Concurrent bulk inserts are safe: they are idempotent by _id.
#   2 CPUs for the Python GIL plus I/O threads. 4G RAM for streaming a JSONL shard.
#   Expected runtime is under 30 min for 130K docs at ~20K docs/s.
#
# Requires:
#   1. ES running (start_es.sbatch)
#   2. The shards $CHUNKS_V3/shards/enriched_ner/enriched_ner_XXXX.jsonl (16 files)
#
# Submit with scripts/pipeline/04_es.sh, which chains the dependencies and the finalize job.

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
CHUNKS_V3="${CHUNKS_DIR:-$PROJECT/data/chunks}"
# Read the enriched shards for a consistent passage count (2,091,427).
# --mode base strips the NER fields.
SHARD_DIR="$CHUNKS_V3/shards/enriched_ner"

SHARD_ID=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")

cd "$PROJECT"

ES_URL_FILE="${ES_URL_FILE:-$PROJECT/src/r2c2_index/slurm/logs/.es_url}"
if [[ ! -f "$ES_URL_FILE" ]]; then
    echo "ERROR: $ES_URL_FILE not found. Is start_es.sbatch running?" >&2
    exit 1
fi
ES_URL=$(cat "$ES_URL_FILE")
echo "Task $SLURM_ARRAY_TASK_ID on $(hostname): shard $SHARD_ID"
echo "Using ES at: $ES_URL"

if ! curl -s "$ES_URL/_cluster/health" | grep -q '"status"'; then
    echo "ERROR: ES not reachable at $ES_URL" >&2
    exit 1
fi

# Task 0 creates the index. The mapping is defined once, in Python.
SENTINEL="${INDEX_SENTINEL_DIR:-$PROJECT/src/r2c2_index/slurm/logs}/.index_created_bitem-passages"
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    echo "Task 0: Creating index bitem-passages..."
    rm -f "$SENTINEL"
    .venv/bin/python3 -u -m r2c2_index.es_index es_manage_index create \
        --es-url "$ES_URL" --index-name bitem-passages --mode base --recreate
    touch "$SENTINEL"
    echo "Index created, sentinel written."
fi
# The other tasks wait for the sentinel that task 0 writes on success.
if [[ "$SLURM_ARRAY_TASK_ID" != "0" ]]; then
    for i in $(seq 1 90); do
        if [[ -f "$SENTINEL" ]]; then break; fi
        if [[ $i -eq 90 ]]; then
            echo "ERROR: Index creation sentinel not found after 180s. Task 0 may have failed." >&2
            exit 1
        fi
        sleep 2
    done
fi

SHARD_FILE="$SHARD_DIR/enriched_ner_${SHARD_ID}.jsonl"
if [[ ! -f "$SHARD_FILE" ]]; then
    echo "ERROR: Shard file not found: $SHARD_FILE" >&2
    exit 1
fi

SHARD_LINES=$(wc -l < "$SHARD_FILE")
echo "Shard $SHARD_ID: $SHARD_LINES lines"

.venv/bin/python3 -u -m r2c2_index.es_index index_to_elasticsearch \
    --input "$SHARD_FILE" \
    --es-url "$ES_URL" \
    --index-name bitem-passages \
    --mode base \
    --shard-mode

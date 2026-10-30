#!/bin/bash
#SBATCH --job-name=r2c2-merge-offsets
#SBATCH --partition=shared-cpu
#SBATCH --array=0-15
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/merge_offsets_%A_%a.out
#SBATCH --error=src/r2c2_index/slurm/logs/merge_offsets_%A_%a.err
#
# Resources:
#   shared-cpu: smallest node has 12 CPUs and 62G RAM; 12h limit.
#   16 parallel tasks. Each loads the full offset dict (~500 MB), then processes
#   a ~375 MB enriched JSONL shard. Peak RAM ~2-3 GB, so 4G is safe.
#   1 CPU: the work is I/O-bound streaming. Expected runtime is under 30 min per shard.
#
# Requires $CHUNKS_DIR/passages_enriched.jsonl split into the 16 shards
#   $CHUNKS_V3/shards/enriched/enriched_XXXX.jsonl

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
CHUNKS_V3="${CHUNKS_DIR:-$PROJECT/data/chunks}"
SHARD_DIR="$CHUNKS_V3/shards/enriched"
OUTPUT_DIR="$CHUNKS_V3/shards/enriched_out"

SHARD_ID=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")

cd "$PROJECT"
mkdir -p "$OUTPUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID on $(hostname): shard $SHARD_ID"

SHARD_FILE="$SHARD_DIR/enriched_${SHARD_ID}.jsonl"
if [[ ! -f "$SHARD_FILE" ]]; then
    echo "ERROR: Shard file not found: $SHARD_FILE" >&2
    exit 1
fi

SHARD_LINES=$(wc -l < "$SHARD_FILE")
echo "Shard $SHARD_ID: $SHARD_LINES lines"

.venv/bin/python3 -u -m r2c2_index.ner merge_offsets_to_enriched \
    --passages-v3 "$CHUNKS_V3/passages.jsonl" \
    --enriched "$SHARD_FILE" \
    --output "$OUTPUT_DIR/enriched_v3_${SHARD_ID}.jsonl" \
    --entity-map "$OUTPUT_DIR/entity_counts_${SHARD_ID}.json" \
    --shard-mode

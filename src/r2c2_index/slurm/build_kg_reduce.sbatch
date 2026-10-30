#!/bin/bash
#SBATCH --job-name=r2c2-kg-reduce
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/build_kg_reduce_%j.out
#SBATCH --error=src/r2c2_index/slurm/logs/build_kg_reduce_%j.err
#
# Reduce phase of the KG build: merge the 16 partial count files, compute NPMI, prune.
# Memory: the full pair-count dictionaries (~16 shards x ~25M pairs each) are held in RAM.

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
PYTHON="$PROJECT/.venv/bin/python3"
CHUNKS_V3="${CHUNKS_DIR:-$PROJECT/data/chunks}"
INDEXES_DIR="${INDEXES_DIR:-$PROJECT/data/indexes}"

cd "$PROJECT"
mkdir -p "$INDEXES_DIR"

echo "=== KG Reduce Phase ==="
echo "Node: $(hostname)"
echo "Start: $(date -Iseconds)"

SHARD_COUNT=$(ls "$CHUNKS_V3/shards/kg_out"/kg_counts_*.json 2>/dev/null | wc -l)
echo "Shard files found: $SHARD_COUNT"
if [[ "$SHARD_COUNT" -eq 0 ]]; then
    echo "ERROR: No KG shard files found in $CHUNKS_V3/shards/kg_out/" >&2
    exit 1
fi

PYTHONUNBUFFERED=1 $PYTHON -u -m r2c2_index.kg_build reduce \
    --input-dir "$CHUNKS_V3/shards/kg_out" \
    --input-pattern "kg_counts_*.json" \
    --output "$INDEXES_DIR/bitem-kg.json"

echo ""
echo "--- KG Validation ---"
PYTHONUNBUFFERED=1 $PYTHON -u -c "
import json, sys
kg = json.load(open('$INDEXES_DIR/bitem-kg.json'))
edges = kg.get('total_edges', 0)
entities = kg.get('total_entities', 0)
print(f'Edges: {edges:,}, Entities: {entities:,}')
if entities < 1000:
    print(f'FAIL: expected >= 1,000 entities, got {entities}')
    sys.exit(1)
if edges < 1000:
    print(f'FAIL: expected >= 1,000 edges, got {edges}')
    sys.exit(1)
print('PASS: KG validation')
"

echo ""
ls -lh "$INDEXES_DIR/bitem-kg.json"
echo "End: $(date -Iseconds)"
echo "=== KG Reduce Complete ==="

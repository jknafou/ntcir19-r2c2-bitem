#!/bin/bash
#SBATCH --job-name=r2c2-faiss-enriched
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/faiss_enriched_%j.out
#SBATCH --error=src/r2c2_index/slurm/logs/faiss_enriched_%j.err
#
# Resources:
#   Same as build_faiss_plain.sbatch. Building the enriched text adds an entity
#   prefix to each passage, and encoding time is similar. The two jobs are
#   submitted independently.
#
# Requires passages_enriched_v3.jsonl and entity_type_map.json.

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
CHUNKS_V3="${CHUNKS_DIR:-$PROJECT/data/chunks}"
INDEXES_DIR="${INDEXES_DIR:-$PROJECT/data/indexes}"

cd "$PROJECT"

echo "=== FAISS Enriched Index Build ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date -Iseconds)"

# Cache the model before encoding starts.
.venv/bin/python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
print('Model cached')
"

.venv/bin/python3 -u -m r2c2_index.faiss_encode build_dense_index \
    --passages "$CHUNKS_V3/passages_enriched_v3.jsonl" \
    --output-dir "$INDEXES_DIR/bitem-faiss-enriched" \
    --model nomic-ai/nomic-embed-text-v1.5 \
    --index-type enriched \
    --entity-type-map "$CHUNKS_V3/entity_type_map.json" \
    --batch-size 1024 \
    --device cuda

# Remove a stale id_map.json if one is present.
rm -f "$INDEXES_DIR/bitem-faiss-enriched/id_map.json"

echo "End: $(date -Iseconds)"

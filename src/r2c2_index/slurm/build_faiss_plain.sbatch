#!/bin/bash
#SBATCH --job-name=r2c2-faiss-plain
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/faiss_plain_%j.out
#SBATCH --error=src/r2c2_index/slurm/logs/faiss_plain_%j.err
#
# Resources:
#   shared-gpu: 12h limit. One GPU of any type (A100, RTX 3090, etc.).
#   8 CPUs for data loading and batching. 64G RAM for 2.09M passage texts and
#   embeddings (768-dim float32 = ~6.4 GB for the final matrix alone, plus text
#   buffers).
#   Encoding 2.09M passages at ~500 passages/s takes ~1.2h on an A100; slower
#   GPUs (Titan X, P100) can take 6-8h.
#
# Requires passages.jsonl.

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
CHUNKS_V3="${CHUNKS_DIR:-$PROJECT/data/chunks}"
INDEXES_DIR="${INDEXES_DIR:-$PROJECT/data/indexes}"

cd "$PROJECT"

echo "=== FAISS Plain Index Build ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date -Iseconds)"

# Cache the model before encoding starts, in case the HF hub is slow.
.venv/bin/python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
print('Model cached')
"

.venv/bin/python3 -u -m r2c2_index.faiss_encode build_dense_index \
    --passages "$CHUNKS_V3/passages.jsonl" \
    --output-dir "$INDEXES_DIR/bitem-faiss-plain" \
    --model nomic-ai/nomic-embed-text-v1.5 \
    --index-type plain \
    --batch-size 1024 \
    --device cuda

# Remove a stale id_map.json if one is present.
rm -f "$INDEXES_DIR/bitem-faiss-plain/id_map.json"

echo "End: $(date -Iseconds)"

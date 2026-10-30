#!/bin/bash
#SBATCH --job-name=r2c2-chunk
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/chunk_%j.out
#SBATCH --error=src/r2c2_index/slurm/logs/chunk_%j.err
#
# Stage 1: parse + chunk the R2C2 wikitext corpus into passages + documents.
#
# Inputs:  $CORPUS_DIR/r2c2_corpus_v1.1/wikitext/{wookieepedia,wikipedia}_wikitext.tar.gz
# Outputs: $CHUNKS_DIR/{passages.jsonl, documents.jsonl, corpus_meta.json}
#
# Required env vars:
#   PROJECT     repo root (defaults to $SLURM_SUBMIT_DIR)
#   CORPUS_DIR  parent of r2c2_corpus_v1.1/ (defaults to $PROJECT/data)
#   CHUNKS_DIR  output dir (defaults to $PROJECT/data/chunks)
#
# Single-process chunker (in-memory, ~1-2 h on the full corpus). The chunker
# itself parallelises within the process; the SBATCH script is not an array.

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
CORPUS_DIR="${CORPUS_DIR:-$PROJECT/data}"
CHUNKS_DIR="${CHUNKS_DIR:-$PROJECT/data/chunks}"

WIKITEXT_DIR="$CORPUS_DIR/r2c2_corpus_v1.1/wikitext"
PYTHON="$PROJECT/.venv/bin/python3"

mkdir -p "$CHUNKS_DIR" src/r2c2_index/slurm/logs

CHUNK_INPUTS=()
for archive in "$WIKITEXT_DIR/wookieepedia_wikitext.tar.gz" "$WIKITEXT_DIR/wikipedia_wikitext.tar.gz"; do
    if [ ! -f "$archive" ]; then
        echo "ERROR: missing input archive: $archive" >&2
        echo "Run scripts/download_corpus.sh first." >&2
        exit 1
    fi
    source_tag="$(basename "${archive%.tar.gz}")"
    CHUNK_INPUTS+=(--input "${archive}:${source_tag}")
done

echo "=== r2c2 chunk on $(hostname) ==="
echo "Start: $(date -Iseconds)"
echo "  PROJECT     = $PROJECT"
echo "  CORPUS_DIR  = $CORPUS_DIR"
echo "  CHUNKS_DIR  = $CHUNKS_DIR"

cd "$PROJECT"

LIMIT_ARGS=()
if [ "$DEMO" = 1 ]; then
    # Reservoir-sample DEMO_N docs uniformly from each archive (seeded
    # by DEMO_SEED — set in _demo_arg.sh, default 42) so the demo
    # sub-corpus is representative rather than first-N-by-archive-order.
    LIMIT_ARGS=(--limit "$DEMO_N" --random-seed "$DEMO_SEED")
    echo "  --limit     = $DEMO_N (demo mode, random-seed=$DEMO_SEED)"
fi

PYTHONUNBUFFERED=1 "$PYTHON" -m r2c2_index.corpus_parse chunk_structured \
    "${CHUNK_INPUTS[@]}" \
    --output-dir "$CHUNKS_DIR" \
    "${LIMIT_ARGS[@]}"

echo "End: $(date -Iseconds)"

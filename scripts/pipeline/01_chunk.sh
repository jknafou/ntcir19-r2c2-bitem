#!/usr/bin/env bash
# Phase 1 — Corpus parse + chunking (single command).
#
# Reads the R2C2 corpus tarballs at
#   $CORPUS_DIR/r2c2_corpus_v1.1/wikitext/{wookieepedia,wikipedia}_wikitext.tar.gz
# and writes ~200-word passages + per-document metadata at
#   $CHUNKS_DIR/{passages,documents,corpus_meta}.jsonl
#
# This is the foundation every later phase reads — NER enriches the
# passages, KG builds entity co-occurrence on the enriched output, ES
# and FAISS index the chunks.
#
# Demo (--demo):
#   The chunker is invoked with `--limit 10000 --random-seed 42`, which
#   reservoir-samples 10K docs per source uniformly over the full
#   archive (seeded for reproducibility). Outputs land in
#   `data/r2c2_corpus_demo_N10000/chunks/` instead of `data/chunks/`.
#   Override the seed with `DEMO_SEED=<int>`.
#
# Usage:
#   bash scripts/pipeline/01_chunk.sh
#   bash scripts/pipeline/01_chunk.sh --demo

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

echo "=== Phase 1: corpus parse + chunking $([ "$DEMO" = 1 ] && echo "(--demo)") ==="

CHUNK_JID=$(sbatch --parsable src/r2c2_index/slurm/run_pipeline.sbatch $DEMO_ARG)
echo "  [1/1] chunker  job=$CHUNK_JID"

echo ""
echo "Final artefacts (when chunker finishes):"
if [ "$DEMO" = 1 ]; then
    echo "  $REPO_ROOT/data/r2c2_corpus_demo_N10000/chunks/{passages,documents,corpus_meta}.jsonl"
else
    echo "  $REPO_ROOT/data/chunks/{passages,documents,corpus_meta}.jsonl"
fi

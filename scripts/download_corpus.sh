#!/usr/bin/env bash
# Download the R2C2 corpus v1.1 from the official Waseda Box shared folder.
#
# Layout produced:
#   <DATA_DIR>/r2c2_corpus_v1.1/
#     wikitext/
#       wikipedia_wikitext.tar.gz       (~1.2 GB — raw articles)
#       wookieepedia_wikitext.tar.gz    (~233 MB — raw articles)
#     baseline_passages/
#       wikipedia_passages.tar.gz       (~825 MB — pre-chunked passages)
#       wookieepedia_passages.tar.gz    (~195 MB — pre-chunked passages)
#
# Usage:
#   bash scripts/download_corpus.sh [DATA_DIR]
#
# DATA_DIR defaults to ./data
# Set FORCE=1 to re-download files that already exist.

set -euo pipefail

DATA_DIR="${1:-./data}"
FORCE="${FORCE:-0}"

# Waseda Box shared folder — verified 2026-05-14
SHARED_NAME="j9aa9j7ztq7wtkcqy73kpxyy7jljha56"
BOX_BASE="https://waseda.app.box.com/index.php?rm=box_download_shared_file"

declare -A FILES=(
    ["r2c2_corpus_v1.1/wikitext/wikipedia_wikitext.tar.gz"]="f_2039454059675"
    ["r2c2_corpus_v1.1/wikitext/wookieepedia_wikitext.tar.gz"]="f_2039453771948"
    ["r2c2_corpus_v1.1/baseline_passages/wikipedia_passages.tar.gz"]="f_2039458662950"
    ["r2c2_corpus_v1.1/baseline_passages/wookieepedia_passages.tar.gz"]="f_2039459925220"
)

echo "Downloading R2C2 corpus v1.1 from Waseda Box into $DATA_DIR/r2c2_corpus_v1.1/"

for rel_path in "${!FILES[@]}"; do
    file_id="${FILES[$rel_path]}"
    dest="$DATA_DIR/$rel_path"
    if [ -f "$dest" ] && [ "$FORCE" != "1" ]; then
        echo "  skip (exists): $rel_path"
        continue
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  downloading $rel_path …"
    curl -fSL --progress-bar \
        "${BOX_BASE}&shared_name=${SHARED_NAME}&file_id=${file_id}" \
        -o "$dest"
    echo "  done: $rel_path"
done

echo "Corpus download complete: $DATA_DIR/r2c2_corpus_v1.1/"

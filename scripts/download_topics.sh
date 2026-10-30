#!/usr/bin/env bash
# Download the R2C2 topics files from the Waseda Box shared links and convert
# them to the JSONL format expected by the inference pipeline.
#
# Files produced:
#   <DATA_DIR>/r2c2/test_questions.jsonl    (65 NTCIR-19 R2C2 test questions)
#   <DATA_DIR>/r2c2/dry_run_questions.jsonl (5 official dry-run questions)
#
# Usage:
#   bash scripts/download_topics.sh [DATA_DIR]
#
# DATA_DIR defaults to ./data
# Set FORCE=1 to overwrite files that already exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-./data}"
FORCE="${FORCE:-0}"

# Waseda Box download parameters (stable shared links, verified 2026-05-13)
TEST_SHARED_NAME="dlzecwijsv6rnoaytv74cjvcraj5ch28"
TEST_FILE_ID="f_2163299132962"
DRYRUN_SHARED_NAME="vi6qxsokhsktt9u4k1ssyun43t0aa4kh"
DRYRUN_FILE_ID="f_1978690132752"

BOX_BASE="https://waseda.app.box.com/index.php?rm=box_download_shared_file"

mkdir -p "$DATA_DIR/r2c2"

_download_and_convert() {
    local label="$1"
    local shared_name="$2"
    local file_id="$3"
    local dest="$4"

    if [ -f "$dest" ] && [ "$FORCE" != "1" ]; then
        echo "  skip (exists): $dest — set FORCE=1 to overwrite"
        return
    fi
    echo "  downloading $label …"
    local url="${BOX_BASE}&shared_name=${shared_name}&file_id=${file_id}"
    curl -fsSL "$url" \
        | python3 "$SCRIPT_DIR/convert_topics_xml_to_jsonl.py" \
        > "$dest"
    local n
    n=$(wc -l < "$dest")
    echo "  wrote $n questions → $dest"
}

echo "Downloading R2C2 topics from Waseda Box into $DATA_DIR/r2c2/"
_download_and_convert "test questions (65 qids)" \
    "$TEST_SHARED_NAME" "$TEST_FILE_ID" \
    "$DATA_DIR/r2c2/test_questions.jsonl"
_download_and_convert "dry-run questions (5 qids)" \
    "$DRYRUN_SHARED_NAME" "$DRYRUN_FILE_ID" \
    "$DATA_DIR/r2c2/dry_run_questions.jsonl"
echo "Topics download complete."

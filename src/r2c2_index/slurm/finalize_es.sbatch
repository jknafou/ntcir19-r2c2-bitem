#!/bin/bash
#SBATCH --job-name=r2c2-es-finalize
#SBATCH --partition=shared-cpu
## #SBATCH --nodelist=cpu330  # uncomment + edit to pin ES traffic to a single CPU node
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/finalize_es_%j.out
#SBATCH --error=src/r2c2_index/slurm/logs/finalize_es_%j.err
#
# Finalize an ES index after parallel bulk indexing: refresh, force-merge, count.
# Must run on the node where ES runs when ES binds to localhost.

set -euo pipefail

INDEX_NAME="${INDEX_NAME:?Set INDEX_NAME via --export=INDEX_NAME=bitem-passages}"

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
PYTHON="$PROJECT/.venv/bin/python3"

ES_URL_FILE="${ES_URL_FILE:-$PROJECT/src/r2c2_index/slurm/logs/.es_url}"
if [[ ! -f "$ES_URL_FILE" ]]; then
    echo "ERROR: $ES_URL_FILE not found. Is start_es.sbatch running?" >&2
    exit 1
fi
ES_URL=$(cat "$ES_URL_FILE")

echo "=== Finalize ES index: $INDEX_NAME ==="
echo "Node: $(hostname)"
echo "ES URL: $ES_URL"

cd "$PROJECT"

PYTHONUNBUFFERED=1 $PYTHON -u -m r2c2_index.es_index es_manage_index finalize \
    --es-url "$ES_URL" \
    --index-name "$INDEX_NAME"

echo ""
echo "--- Doc-count validation ---"
PYTHONUNBUFFERED=1 $PYTHON -u -c "
from elasticsearch import Elasticsearch
import sys
es = Elasticsearch('$ES_URL')
count = es.count(index='$INDEX_NAME')['count']
print(f'Index $INDEX_NAME: {count:,} docs')

EXPECTED = {
    'bitem-passages': 1950000,
    'bitem-passages-ner': 2090000,
    'bitem-documents': 480000,
}
min_count = EXPECTED.get('$INDEX_NAME', 0)
if min_count > 0 and count < min_count:
    print(f'FAIL: expected >= {min_count:,}, got {count:,}')
    sys.exit(1)
print(f'PASS: doc count >= {min_count:,}')
"

echo ""
echo "=== Finalize complete: $INDEX_NAME ==="

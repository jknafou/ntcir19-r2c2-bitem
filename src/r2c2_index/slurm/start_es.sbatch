#!/bin/bash
#SBATCH --job-name=r2c2-elasticsearch
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_index/slurm/logs/elasticsearch_%j.out
#SBATCH --error=src/r2c2_index/slurm/logs/elasticsearch_%j.err
#
# Resources: shared-cpu nodes have 12-36 CPUs and 62-187G RAM, with a 12 h
#   limit. 48G covers the ES heap plus OS overhead; 4 CPUs cover GC and
#   indexing. 12 h is the partition limit, so ES stays up for one indexing
#   window; resubmit if indexing needs longer.
#
# This is a long-running service job. Indexing jobs connect to ES on this
# node, so the hostname is printed at startup for them to target.

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
ES_DATA="${ES_DATA_DIR:-$PROJECT/data/es-data}"
ES_VERSION="8.17.0"

mkdir -p "$ES_DATA"

echo "=== Elasticsearch $ES_VERSION ==="
echo "Node: $(hostname)"
echo "ES data: $ES_DATA"
echo "Start: $(date -Iseconds)"
echo ""
echo "IMPORTANT: ES binds to localhost only (no root for vm.max_map_count)."
echo "Indexing jobs MUST run on this same node: --nodelist=$(hostname)"
echo "ES URL: http://localhost:9200"
echo ""

# Pull image if needed (apptainer caches in ~/.apptainer)
# ES 8.17 settings passed via -E flags (env vars with dots are unreliable)
apptainer exec \
    --writable-tmpfs \
    --bind "$ES_DATA:/usr/share/elasticsearch/data" \
    --env ES_JAVA_OPTS="-Xms16g -Xmx16g" \
    "docker://docker.elastic.co/elasticsearch/elasticsearch:$ES_VERSION" \
    elasticsearch \
        -Ediscovery.type=single-node \
        -Expack.security.enabled=false \
        -Enetwork.host=_local_ \
        -Ehttp.host=_local_ &

ES_PID=$!

echo "Waiting for ES to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:9200/_cluster/health" 2>/dev/null | grep -q '"status"'; then
        echo "ES is ready ($(date -Iseconds))"
        curl -s "http://localhost:9200" | python3 -m json.tool
        break
    fi
    sleep 2
done

# Delete old test indexes only (preserve production bitem-* indexes)
echo ""
echo "Cleaning up old test/r2c2 indexes..."
curl -X DELETE "http://localhost:9200/r2c2-*" 2>/dev/null || true
curl -X DELETE "http://localhost:9200/bitem-test-*" 2>/dev/null || true
echo "NOTE: Production bitem-* indexes preserved. Delete manually if rebuild needed."
echo ""

# Write connection info for downstream jobs (URL + node name for --nodelist)
ES_URL_FILE="${ES_URL_FILE:-$PROJECT/src/r2c2_index/slurm/logs/.es_url}"
ES_NODE_FILE="${ES_NODE_FILE:-$PROJECT/src/r2c2_index/slurm/logs/.es_node}"
mkdir -p "$(dirname "$ES_URL_FILE")" "$(dirname "$ES_NODE_FILE")"
echo "http://localhost:9200" > "$ES_URL_FILE"
echo "$(hostname)" > "$ES_NODE_FILE"
echo "ES URL written to $ES_URL_FILE"
echo "ES node written to $ES_NODE_FILE"

# Keep running until SLURM kills us
echo "ES running on PID $ES_PID. Waiting..."
wait $ES_PID

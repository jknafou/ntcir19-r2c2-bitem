#!/bin/bash
# Worker sbatch: starts ES + MCP + vLLM + worker HTTP server on a GPU node.
#
# Environment variables (set by orchestrator):
#   MODEL          — HF model ID (default: Qwen/Qwen3.5-122B-A10B-FP8)
#   ADAPTER        — optional LoRA adapter path
#   WORKER_PORT    — HTTP server port (default: auto from job ID)
#   ES_DATA_DIR    — Elasticsearch data dir (required; produced by Stage 5)
#   ES_SIF_PATH    — Apptainer image (.sif). Falls back to docker://docker.elastic.co/elasticsearch/elasticsearch:8.17.0
#   PRIMARY_CLUSTER — orchestrator's cluster name (auto-detected via env)
#
# Submit via the orchestrator (do not invoke directly).

#SBATCH --job-name=orch-worker
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --output=src/r2c2_agent/slurm/logs/worker_%j.out
#SBATCH --error=src/r2c2_agent/slurm/logs/worker_%j.err

set -euo pipefail

PROJECT="${PROJECT:-$(pwd)}"
MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
ADAPTER="${ADAPTER:-}"

JOB_PORT_BASE=$((10000 + (${SLURM_JOB_ID:-$$} % 13800) * 4))
VLLM_PORT=$((JOB_PORT_BASE))
LOCAL_MCP_PORT=$((JOB_PORT_BASE + 1))
ES_PORT=$((JOB_PORT_BASE + 2))
WORKER_PORT="${WORKER_PORT:-$((JOB_PORT_BASE + 3))}"

HOSTNAME_FULL=$(hostname)
if echo "$HOSTNAME_FULL" | grep -qi "bamboo"; then
    CLUSTER="bamboo"
else
    CLUSTER="baobab"
fi

PYTHON="$PROJECT/.venv/bin/python3"
ENDPOINTS_DIR="$PROJECT/src/r2c2_agent/slurm/endpoints"
LOGS_DIR="$PROJECT/src/r2c2_agent/slurm/logs"
mkdir -p "$ENDPOINTS_DIR" "$LOGS_DIR"

echo "=== Orchestrator Worker ==="
echo "  Job:     $SLURM_JOB_ID"
echo "  Cluster: $CLUSTER"
echo "  Node:    $HOSTNAME_FULL"
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
echo "  GPUs:    $(echo "$GPU_INFO" | tr '\n' ', ')"
echo "  Model:   $MODEL"
echo "  Adapter: ${ADAPTER:-none}"
echo "  Ports:   vLLM=$VLLM_PORT MCP=$LOCAL_MCP_PORT ES=$ES_PORT Worker=$WORKER_PORT"

VLLM_PID=""
MCP_PID=""
ES_PID=""
WORKER_PID=""
VERIFIER_SSH_PID=""

cleanup() {
    [ -n "$WORKER_PID" ] && kill $WORKER_PID 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && kill $TUNNEL_PID 2>/dev/null || true
    [ -n "$VERIFIER_SSH_PID" ] && kill $VERIFIER_SSH_PID 2>/dev/null || true
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true
    [ -n "$MCP_PID" ] && kill $MCP_PID 2>/dev/null || true
    [ -n "${NLI_PID:-}" ] && kill $NLI_PID 2>/dev/null || true
    [ -n "${DEBERTA_PID:-}" ] && kill $DEBERTA_PID 2>/dev/null || true
    [ -n "$ES_PID" ] && kill $ES_PID 2>/dev/null || true
    [ -n "${ES_DATA_LOCAL:-}" ] && rm -rf "$ES_DATA_LOCAL" 2>/dev/null || true
    rm -f "$ENDPOINTS_DIR/worker_${SLURM_JOB_ID}.endpoint" 2>/dev/null || true
    [ "${CLUSTER:-}" = "bamboo" ] && ssh -o ConnectTimeout=5 baobab \
        "rm -f $ENDPOINTS_DIR/worker_${SLURM_JOB_ID}.endpoint" 2>/dev/null || true
}
trap cleanup EXIT

# Start Elasticsearch
echo ""
echo "--- Starting Elasticsearch ---"
ES_DATA_DIR="${ES_DATA_DIR:?ES_DATA_DIR must be set (output of src/r2c2_index/slurm/finalize_es.sbatch)}"
ES_SIF_PATH="${ES_SIF_PATH:-$PROJECT/containers/elasticsearch-8.17.0.sif}"
ES_DOCKER="docker://docker.elastic.co/elasticsearch/elasticsearch:8.17.0"
ES_IMAGE="$ES_SIF_PATH"
[ ! -f "$ES_SIF_PATH" ] && ES_IMAGE="$ES_DOCKER"

if [ ! -d "$ES_DATA_DIR" ] || [ -z "$(ls -A "$ES_DATA_DIR" 2>/dev/null)" ]; then
    echo "ERROR: ES data not found at $ES_DATA_DIR"
    ES_BROKEN=1
fi

ES_DATA_LOCAL="/tmp/es-data-${SLURM_JOB_ID:-$$}"
echo "  Copying ES data to $ES_DATA_LOCAL..."
cp -a "$ES_DATA_DIR" "$ES_DATA_LOCAL"

apptainer run --writable-tmpfs --no-home \
    --env "ES_JAVA_OPTS=-Xms4g -Xmx4g" \
    --bind "$ES_DATA_LOCAL:/usr/share/elasticsearch/data" \
    "$ES_IMAGE" \
    elasticsearch \
        -Ediscovery.type=single-node \
        -Expack.security.enabled=false \
        -Enetwork.host=_local_ \
        -Ehttp.host=_local_ \
        -Ehttp.port=$ES_PORT &
ES_PID=$!

for i in $(seq 1 120); do
    curl -sf "http://localhost:${ES_PORT}/_cluster/health" >/dev/null 2>&1 && break
    if ! kill -0 $ES_PID 2>/dev/null; then
        echo "ERROR: ES died during startup"
        ES_BROKEN=1
        break
    fi
    sleep 2
done

# Start MCP
echo ""
echo "--- Starting MCP ---"
cd "$PROJECT"
MCP_LOG="$LOGS_DIR/mcp_${SLURM_JOB_ID}.log"
ELASTICSEARCH_URL="http://localhost:${ES_PORT}" \
MCP_PORT="$LOCAL_MCP_PORT" \
PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 "$PYTHON" -m r2c2_agent.mcp_server.server --server 2>&1 | tee "$MCP_LOG" &
MCP_PID=$!

for i in $(seq 1 120); do
    if grep -q "Application startup complete" "$MCP_LOG" 2>/dev/null; then
        echo "  MCP ready after $((i * 3))s"
        break
    fi
    if ! kill -0 $MCP_PID 2>/dev/null; then
        echo "ERROR: MCP died during startup"
        MCP_BROKEN=1
        break
    fi
    sleep 3
done
MCP_URL="http://localhost:${LOCAL_MCP_PORT}/mcp"

# Start NLI oracle (MiniCheck-Flan-T5-Large on CPU)
NLI_PID=""
NLI_PORT="${NLI_PORT:-18500}"
echo ""
echo "--- Starting NLI oracle (MiniCheck-Flan-T5-Large on CPU) ---"
NLI_LOG="$LOGS_DIR/nli_${SLURM_JOB_ID}.log"
PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 "$PYTHON" -m r2c2_agent.orchestrator.nli_verifier \
    --port "$NLI_PORT" 2>&1 | tee "$NLI_LOG" &
NLI_PID=$!
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${NLI_PORT}/health" >/dev/null 2>&1; then
        echo "  NLI ready after $((i * 3))s"
        break
    fi
    if ! kill -0 $NLI_PID 2>/dev/null; then echo "WARN: NLI died"; NLI_PID=""; break; fi
    sleep 3
done
[ -n "$NLI_PID" ] && export NLI_SERVER_URL="http://localhost:${NLI_PORT}"

# Start DeBERTa-v3-MNLI (CPU hybrid-cascade tiebreak)
DEBERTA_PID=""
DEBERTA_PORT="${DEBERTA_PORT:-18600}"
echo ""
echo "--- Starting DeBERTa-v3-MNLI ---"
DEBERTA_LOG="$LOGS_DIR/deberta_${SLURM_JOB_ID}.log"
PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 "$PYTHON" -m r2c2_agent.orchestrator.nli_verifier \
    --backend deberta_mnli \
    --port "$DEBERTA_PORT" 2>&1 | tee "$DEBERTA_LOG" &
DEBERTA_PID=$!
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${DEBERTA_PORT}/health" >/dev/null 2>&1; then
        echo "  DeBERTa ready after $((i * 3))s"
        break
    fi
    if ! kill -0 $DEBERTA_PID 2>/dev/null; then DEBERTA_PID=""; break; fi
    sleep 3
done
[ -n "$DEBERTA_PID" ] && export DEBERTA_NLI_URL="http://localhost:${DEBERTA_PORT}"

# Start vLLM
echo ""
TP_SIZE="${SLURM_GPUS_ON_NODE:-2}"
echo "--- Starting vLLM (TP=$TP_SIZE) ---"

STALE_PID=$(lsof -ti tcp:$VLLM_PORT 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 2
fi

GPU_NAME_FIRST=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)
if echo "$GPU_NAME_FIRST" | grep -qiE 'H100|H200|B200|Blackwell'; then
    export NCCL_NET_GDR_LEVEL=0
    export GLOO_SOCKET_IFNAME=lo
else
    export NCCL_SHM_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export GLOO_SOCKET_IFNAME=lo
fi

GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
GPU_MEM_MB="${GPU_MEM_MB:-0}"
if [ -n "${MAX_MODEL_LEN:-}" ]; then
    echo "  Using orchestrator override: MAX_MODEL_LEN=$MAX_MODEL_LEN"
else
    if [ "${GPU_MEM_MB:-0}" -ge 120000 ]; then
        MAX_MODEL_LEN=131072
    elif [ "${GPU_MEM_MB:-0}" -ge 90000 ]; then
        MAX_MODEL_LEN=81920
    else
        MAX_MODEL_LEN=40960
    fi
fi
CONTEXT_BUDGET=$((MAX_MODEL_LEN * 95 / 100))

VLLM_EXTRA_ARGS=""
if [ -n "$ADAPTER" ]; then
    VLLM_EXTRA_ARGS="--enable-lora --lora-modules adapter=$ADAPTER"
fi

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enforce-eager \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    $VLLM_EXTRA_ARGS &
VLLM_PID=$!

i=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready after $((i * 2))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died"
        VLLM_BROKEN=1
        break
    fi
    sleep 2
    i=$((i + 1))
done

# CP2 verifier (external sbatch; optional)
VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
VERIFIER_URL=""
PROJECT_BAOBAB="${PROJECT_BAOBAB:-$PROJECT}"
VERIFIER_EP_FILE="$PROJECT_BAOBAB/src/r2c2_agent/slurm/endpoints/verifier.endpoint"
if [ -f "$VERIFIER_EP_FILE" ]; then
    EP_JSON=$(cat "$VERIFIER_EP_FILE" 2>/dev/null)
    VHOST=$(echo "$EP_JSON" | "$PYTHON" -c "import json,sys; print(json.loads(sys.stdin.read()).get('host',''))" 2>/dev/null)
    VPORT=$(echo "$EP_JSON" | "$PYTHON" -c "import json,sys; print(json.loads(sys.stdin.read()).get('port',''))" 2>/dev/null)
    if [ -n "$VHOST" ] && [ -n "$VPORT" ]; then
        VERIFIER_URL="http://${VHOST}:${VPORT}/v1"
        echo "  CP2 verifier: $VERIFIER_URL"
    fi
fi

GPU_SHORT=$(echo "$GPU_NAME_FIRST" | sed 's/ /-/g')
export GPU_INFO="$GPU_SHORT"

# Start worker HTTP server
echo ""
echo "--- Starting worker HTTP server on port $WORKER_PORT ---"

LLM_URL="http://localhost:${VLLM_PORT}/v1" \
MCP_URL="$MCP_URL" \
WORKER_PORT="$WORKER_PORT" \
CONTEXT_BUDGET="$CONTEXT_BUDGET" \
TOKEN_BUDGET="${TOKEN_BUDGET:-$CONTEXT_BUDGET}" \
VERIFIER_URL="${VERIFIER_URL:-}" \
VERIFIER_MODEL="$VERIFIER_MODEL" \
PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
"$PYTHON" -m r2c2_agent.orchestrator.worker &
WORKER_PID=$!

for i in $(seq 1 60); do
    if curl -sf "http://localhost:${WORKER_PORT}/health" >/dev/null 2>&1; then
        echo "Worker ready after $((i * 2))s"
        break
    fi
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "ERROR: Worker HTTP server died"
        break
    fi
    sleep 2
done

# Cross-cluster registration
PRIMARY_CLUSTER="${PRIMARY_CLUSTER:-baobab}"
TUNNEL_PID=""
if [ "$CLUSTER" != "$PRIMARY_CLUSTER" ]; then
    TUNNEL_PORT="${TUNNEL_PORT:-$((19000 + (SLURM_JOB_ID % 1000)))}"
    ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -N -R "${TUNNEL_PORT}:localhost:${WORKER_PORT}" "$PRIMARY_CLUSTER" &
    TUNNEL_PID=$!
    sleep 2
    if kill -0 $TUNNEL_PID 2>/dev/null; then
        ENDPOINT_HOST="localhost"
        ENDPOINT_PORT="$TUNNEL_PORT"
    else
        ENDPOINT_HOST="$HOSTNAME_FULL"
        ENDPOINT_PORT="$WORKER_PORT"
    fi
else
    ENDPOINT_HOST="$HOSTNAME_FULL"
    ENDPOINT_PORT="$WORKER_PORT"
fi

ENDPOINT_FILE="$ENDPOINTS_DIR/worker_${SLURM_JOB_ID}.endpoint"
cat > "$ENDPOINT_FILE" << EOF
{"host": "$ENDPOINT_HOST", "port": $ENDPOINT_PORT, "cluster": "$CLUSTER", "gpu": "$GPU_SHORT", "model": "$MODEL", "job_id": $SLURM_JOB_ID, "status": "ready"}
EOF

if [ "$CLUSTER" != "$PRIMARY_CLUSTER" ]; then
    scp -o StrictHostKeyChecking=no "$ENDPOINT_FILE" \
        "$PRIMARY_CLUSTER:$ENDPOINTS_DIR/" 2>/dev/null && \
        echo "  Endpoint synced to $PRIMARY_CLUSTER" || \
        echo "  WARNING: Could not sync endpoint to $PRIMARY_CLUSTER"
fi

echo "=== Worker $SLURM_JOB_ID ready on $ENDPOINT_HOST:$ENDPOINT_PORT ==="

sleep 43200 &
SLEEP_PID=$!
wait $SLEEP_PID

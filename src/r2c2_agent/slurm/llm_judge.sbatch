#!/bin/bash
# AC-4 LLM-as-Judge sbatch: spins up a vLLM-only worker (no MCP/ES/NLI/DeBERTa)
# for the ~65-call AC-4 judge pass. Writes an endpoint file consumed by
# `r2c2_agent.orchestrator.llm_judge`.
#
# Defaults match the production rollout worker (Qwen3.5-122B-A10B-FP8 on
# 2× A100 80 GB with TP=2 — see src/r2c2_agent/slurm/worker.sbatch). Boots in
# ~10-15 min on A100 80GB.
#
# Environment variables:
#   MODEL          — HF model ID (default: Qwen/Qwen3.5-122B-A10B-FP8)
#
# Submit directly:
#   sbatch src/r2c2_agent/slurm/llm_judge.sbatch
# or via:
#   MODEL=... sbatch src/r2c2_agent/slurm/llm_judge.sbatch

#SBATCH --job-name=llm-judge
#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
# Accept any of the three GPU types that have enough VRAM for the
# 122B-FP8 teacher at TP=2 (A100/H100/H200, combined ≥160GB).
# This roughly triples the pool of eligible nodes compared with pinning
# to A100, so the job does not wait in (Resources) while other GPU types
# are free. The bw6k Blackwell type is excluded: 2×48GB is tight for
# this model, and FP8 stalls on those nodes.
#SBATCH --constraint="nvidia_a100_80gb_pcie|nvidia_h100_nvl|nvidia_h200_nvl"
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=src/r2c2_agent/slurm/logs/llm_judge_%j.out
#SBATCH --error=src/r2c2_agent/slurm/logs/llm_judge_%j.err

set -euo pipefail

PROJECT="${PROJECT:-$(pwd)}"

# --demo: parsed for symmetry with the other sbatch templates. Demo
# uses the same model and hardware as production; only the data volume
# differs (Stages 1-5 build a 10K-doc sub-corpus; Stage 6 uses the
# dry-run questions JSONL).
DEMO=0
for arg in "$@"; do case "$arg" in --demo) DEMO=1 ;; esac; done
export DEMO

# Production model: Qwen3.5-122B-A10B-FP8 on 2× A100 80GB via TP=2.
# Override via `MODEL=...` only if you have a specific reason.
MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"

JOB_PORT_BASE=$((11000 + (${SLURM_JOB_ID:-$$} % 1000)))
VLLM_PORT=$((JOB_PORT_BASE))

HOSTNAME_FULL=$(hostname)
# `uv sync` builds a single project venv (Python 3.12, vLLM + CUDA stack and
# the dependencies of every stage) and the worker serves the model from it
# directly. VLLM_PYTHON selects a different interpreter; the default is the
# project `.venv`.
if [ -n "${VLLM_PYTHON:-}" ] && [ -x "$VLLM_PYTHON" ]; then
    PYTHON="$VLLM_PYTHON"
elif [ -x "$PROJECT/.venv/bin/python3" ]; then
    PYTHON="$PROJECT/.venv/bin/python3"
else
    PYTHON=python3
fi
# flashinfer JIT-compiles the gated-delta-net path on first use, shelling out
# to `ninja` (which ships in the venv's bin/). Launching python by full path
# does not put that bin on PATH, so expose it; otherwise the JIT build dies
# with FileNotFoundError: 'ninja' mid-inference.
export PATH="$(dirname "$PYTHON"):$PATH"
ENDPOINTS_DIR="$PROJECT/src/r2c2_agent/slurm/endpoints"
LOGS_DIR="$PROJECT/src/r2c2_agent/slurm/logs"
mkdir -p "$ENDPOINTS_DIR" "$LOGS_DIR"

echo "=== LLM Judge vLLM-only Worker ==="
echo "  Job:     $SLURM_JOB_ID"
echo "  Node:    $HOSTNAME_FULL"
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
echo "  GPUs:    $(echo "$GPU_INFO" | tr '\n' ', ')"
echo "  Model:   $MODEL"
echo "  Port:    vLLM=$VLLM_PORT"

VLLM_PID=""
cleanup() {
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null || true
    rm -f "$ENDPOINTS_DIR/llm_judge_${SLURM_JOB_ID}.endpoint" 2>/dev/null || true
}
trap cleanup EXIT

# Free stale port (idempotent).
STALE_PID=$(lsof -ti tcp:$VLLM_PORT 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 2
fi

GPU_NAME_FIRST=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo "")
if echo "$GPU_NAME_FIRST" | grep -qiE 'H100|H200|B200|Blackwell'; then
    export NCCL_NET_GDR_LEVEL=0
    export GLOO_SOCKET_IFNAME=lo
else
    export NCCL_SHM_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export GLOO_SOCKET_IFNAME=lo
fi

# Qwen3.5-122B-A10B-FP8 supports up to 131072; we cap at 40960 to match
# production rollouts (worker.sbatch picks 40960 when GPU mem < 90GB and
# defaults to TP=2 on 2× A100 80GB). Override via MAX_MODEL_LEN=... .
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"

TP_SIZE="${SLURM_GPUS_ON_NODE:-2}"

echo "--- Starting vLLM (TP=$TP_SIZE, max-model-len=$MAX_MODEL_LEN) ---"

# Tool-call parser selection: production worker uses `qwen3_coder` for
# Qwen3.5-122B-A10B and `hermes` for Qwen3-32B. Default by model name; can
# be overridden via TOOL_CALL_PARSER=... .
if [ -z "${TOOL_CALL_PARSER:-}" ]; then
    if echo "$MODEL" | grep -qiE 'Qwen3\.5.*122B|A10B'; then
        TOOL_CALL_PARSER="qwen3_coder"
    else
        TOOL_CALL_PARSER="hermes"
    fi
fi
echo "  Tool-call parser: $TOOL_CALL_PARSER"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --enable-auto-tool-choice \
    --enforce-eager \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" &
VLLM_PID=$!

i=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready after $((i * 2))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died"
        exit 1
    fi
    sleep 2
    i=$((i + 1))
    if [ $i -gt 1800 ]; then
        echo "ERROR: vLLM did not come up after 60 min"
        exit 1
    fi
done

ENDPOINT_FILE="$ENDPOINTS_DIR/llm_judge_${SLURM_JOB_ID}.endpoint"
cat > "$ENDPOINT_FILE" << EOF
{"host": "$HOSTNAME_FULL", "port": $VLLM_PORT, "url": "http://${HOSTNAME_FULL}:${VLLM_PORT}/v1", "model": "$MODEL", "job_id": $SLURM_JOB_ID, "status": "ready"}
EOF

echo "Endpoint registered at $ENDPOINT_FILE"
echo "URL: http://${HOSTNAME_FULL}:${VLLM_PORT}/v1"

# Hold the job alive while users hit the endpoint.
wait $VLLM_PID

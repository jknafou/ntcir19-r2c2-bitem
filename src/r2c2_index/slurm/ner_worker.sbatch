#!/usr/bin/env bash
# NER work-queue GPU worker (one SLURM array task == one GPU == one vLLM replica).
#
# Each array task:
#
#   1. launches a vLLM server (Qwen/Qwen3.5-9B, TP=1, one replica per GPU),
#   2. publishes its endpoint URL to $WQ_DIR/endpoints/ once /v1/models is up,
#   3. runs the claim->process->mark loop itself: it atomically claims the next
#      pending shard for the current stage (file-rename lock), runs the NER tool
#      against its own local vLLM endpoint, then marks the shard done.
#
#   Each GPU job is self-contained and talks only to its localhost vLLM. The
#   orchestrator only sequences stages and populates queues; the GPUs drain
#   them. This avoids any GPU<->CPU VLAN dependency and works identically on
#   baobab and bamboo.
#
# The stage is read from $WQ_DIR/queue/CURRENT_STAGE (written by the
# orchestrator). Workers idle-wait while no stage is active and exit when the
# orchestrator writes "DONE".
#
# All runtime config arrives through the environment (WQ_DIR, exported via
# --export by the driver) and $WQ_DIR/config. The OR-constraint on GPU type
# lets the same file run on either cluster: SLURM matches only the features
# present locally.
#
#SBATCH --job-name=r2c2-ner-wq
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="nvidia_a100_80gb_pcie|nvidia_h100_nvl|nvidia_h200_nvl|nvidia_rtx_pro_6000_blackwell"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=src/r2c2_index/slurm/logs/ner_wq_worker_%A_%a.out
#SBATCH --error=src/r2c2_index/slurm/logs/ner_wq_worker_%A_%a.err

set -euo pipefail

# WQ_DIR is exported by the driver; everything else comes from $WQ_DIR/config.
WQ_DIR="${WQ_DIR:?WQ_DIR must be set (exported by ner_workqueue.sh)}"
source "$WQ_DIR/config"

# vLLM serves Qwen3.5-9B via a gated-delta-net (linear-attention) path that
# flashinfer JIT-compiles on first use, shelling out to `ninja` (and nvcc).
# `ninja` ships in the venv's bin/, but launching python by full path does
# not put that bin on PATH. Expose it, as activating the venv would; otherwise
# the JIT build dies with FileNotFoundError: 'ninja' mid-inference.
export PATH="$(dirname "$PYTHON"):$PATH"

# Triton allocator for the gated-delta-net (GDN) kernel on Blackwell (sm_120)
# and other non-Hopper GPUs. vLLM only routes that kernel through FlashInfer on
# sm_90 (H100/H200); elsewhere it uses the fla triton TMA path, which needs a
# Triton allocator vLLM never sets for a dense model, and inference crashes
# with "no allocator set". The sitecustomize.py here installs it on the first
# GDN forward; CPython auto-imports it (including in the EngineCore subprocess)
# because its dir is on PYTHONPATH. See src/r2c2_index/ner/_vllm_gdn_fix/sitecustomize.py.
export PYTHONPATH="$REPO_ROOT/src/r2c2_index/ner/_vllm_gdn_fix:${PYTHONPATH:-}"

PROJECT="${PROJECT:-$REPO_ROOT}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}"

log() { echo "[$(date '+%H:%M:%S')] [wq-worker:${JOB_ID}_${TASK_ID}] $*"; }

FD_HARD=$(ulimit -Hn)
if [ "$FD_HARD" -ge 65536 ]; then ulimit -n 65536; else ulimit -n "$FD_HARD"; fi

# Per-task scratch caches avoid NFS contention between concurrent vLLM replicas
# that share a node.
export TORCHDYNAMO_DISABLE=1
export TRITON_CACHE_DIR="/tmp/triton_${JOB_ID}_${TASK_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${JOB_ID}_${TASK_ID}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_${JOB_ID}_${TASK_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# The port is hashed from job_id + task_id so tasks sharing a node do not collide.
PORT=$((10000 + (JOB_ID * 7 + TASK_ID) % 50000))
VLLM_URL="http://localhost:${PORT}/v1"
VLLM_LOG="$WQ_DIR/output/logs/vllm_${JOB_ID}_${TASK_ID}.log"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT SET}  GPU=$GPU_NAME  port=$PORT"

# NCCL/GLOO env. Mostly inert on a single GPU at TP=1; the branch sets the
# P2P/SHM/GDR flags that suit each architecture.
if echo "$GPU_NAME" | grep -qiE 'H100|H200|B200|Blackwell'; then
    export NCCL_NET_GDR_LEVEL=0
    export GLOO_SOCKET_IFNAME=lo
else
    export NCCL_SHM_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export GLOO_SOCKET_IFNAME=lo
fi

# Cleanup: kill vLLM, remove the endpoint, and requeue any shard this worker
# still holds in *_running so a fresh worker or a resume can reclaim it.
VLLM_PID=""
ENDPOINT_FILE="$WQ_DIR/endpoints/worker_${JOB_ID}_${TASK_ID}.endpoint"
CLAIMED_FILE=""
CLAIMED_STAGE=""
cleanup() {
    log "Cleaning up..."
    rm -f "$ENDPOINT_FILE"
    # Requeue an in-flight shard (atomic mv back to pending for its stage).
    if [ -n "$CLAIMED_FILE" ] && [ -n "$CLAIMED_STAGE" ] && [ -f "$CLAIMED_FILE" ]; then
        mv "$CLAIMED_FILE" "$WQ_DIR/queue/${CLAIMED_STAGE}_pending/$(basename "$CLAIMED_FILE")" 2>/dev/null || true
    fi
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    pkill -9 -f "vllm.*--port $PORT" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT
trap 'log "USR1 — walltime, exiting"; exit 0' USR1

# Stagger model loads across tasks on the same node (NFS protection).
sleep $((TASK_ID * 15))

# Pre-flight: warm the BeeGFS metadata cache for the venv before launching vLLM.
# Many workers importing the same site-packages concurrently from BeeGFS can hit
# a transient FileNotFoundError/ImportError on a small-file metadata race. A
# retried warm-up import populates the cache so the vLLM and EngineCore imports
# do not hit it. A transient miss is retried, not fatal.
for attempt in 1 2 3 4 5; do
    if "$PYTHON" -c "import vllm, openai" >/dev/null 2>&1; then
        log "pre-flight import OK (attempt $attempt) — venv metadata warm"
        break
    fi
    log "pre-flight import flaked (attempt $attempt/5, BeeGFS small-file race) — retrying"
    sleep $((attempt * 5))
done

log "Starting vLLM $MODEL_ID on port $PORT ..."
"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enforce-eager \
    --language-model-only \
    --max-num-batched-tokens 4096 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Poll /v1/models until ready; fail fast if vLLM dies or never comes up.
MAX_WAIT=900; ELAPSED=0
while true; do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "ERROR: vLLM crashed during startup. Last 60 lines:"; tail -60 "$VLLM_LOG"; exit 1
    fi
    sleep 5; ELAPSED=$((ELAPSED + 5))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        log "ERROR: vLLM not ready after ${MAX_WAIT}s. Last 60 lines:"; tail -60 "$VLLM_LOG"; exit 1
    fi
done
log "vLLM healthy after ${ELAPSED}s"
echo "$VLLM_URL" > "$ENDPOINT_FILE"

# The prompt directory is optional.
PROMPT_ARGS=()
if [ -n "${PROMPT_DIR:-}" ] && [ -d "${PROMPT_DIR:-}" ]; then
    PROMPT_ARGS=(--prompt-dir "$PROMPT_DIR")
fi

run() { PYTHONPATH="$REPO_ROOT/src" PYTHONUNBUFFERED=1 "$PYTHON" -m "$@"; }

# process_shard: run the NER tool for the given stage on one claimed shard,
# writing into the shard's own output dir.
process_shard() {
    local stage="$1" sname="$2"
    local shard_input="$WQ_DIR/shards/${sname}.jsonl"
    local sdir="$WQ_DIR/output/shards/$sname"
    mkdir -p "$sdir"

    case "$stage" in
        extract)
            run r2c2_index.ner extract_entities_llm \
                --input "$shard_input" \
                --output "$sdir/extraction.jsonl" \
                --model "$MODEL_ID" \
                --no-propagate \
                --document-level \
                --documents "$DOCUMENTS_JSONL" \
                --vllm-urls "$VLLM_URL" \
                --resume \
                "${PROMPT_ARGS[@]}" \
                >> "$WQ_DIR/output/logs/worker_extract_${JOB_ID}_${TASK_ID}.log" 2>&1
            ;;
        refine)
            run r2c2_index.ner refine_entities \
                --input "$sdir/extraction.jsonl" \
                --output "$sdir/refined.jsonl" \
                --model "$MODEL_ID" \
                --documents "$DOCUMENTS_JSONL" \
                --vllm-urls "$VLLM_URL" \
                --resume \
                "${PROMPT_ARGS[@]}" \
                >> "$WQ_DIR/output/logs/worker_refine_${JOB_ID}_${TASK_ID}.log" 2>&1
            ;;
        crossdoc_review)
            run r2c2_index.ner refine_entities \
                --input "$sdir/refined.jsonl" \
                --output "$sdir/refined_crossdoc.jsonl" \
                --model "$MODEL_ID" \
                --documents "$DOCUMENTS_JSONL" \
                --vllm-urls "$VLLM_URL" \
                --cross-document \
                --precomputed-crossdoc "$WQ_DIR/output/crossdoc_candidates.json" \
                --resume \
                "${PROMPT_ARGS[@]}" \
                >> "$WQ_DIR/output/logs/worker_crossdoc_${JOB_ID}_${TASK_ID}.log" 2>&1
            ;;
        *)
            log "ERROR: unknown stage '$stage'"; return 1 ;;
    esac

    # Validate before marking done (extract only). The extract tool returns [] and
    # exits 0 even when every request fails (vLLM unreachable or crashed). Without
    # this check the claim loop would mark the shard done with zero entities, and
    # a later --resume would skip it forever, silently leaving it uncovered. A
    # real extraction always yields entities on many passages, so a shard with no
    # entities at all fails here and the claim loop re-queues it.
    if [ "$stage" = extract ]; then
        local ne
        ne=$("$PYTHON" -c 'import json,sys
n=0
for line in open(sys.argv[1]):
    if json.loads(line).get("entities"):
        n=1; break
print(n)' "$sdir/extraction.jsonl" 2>/dev/null || echo 0)
        if [ "${ne:-0}" -eq 0 ]; then
            log "[extract] $sname: 0 entities across the whole shard (vLLM likely down) — failing for re-queue"
            return 1
        fi
    fi
    return 0
}

# Main claim loop. The orchestrator publishes the active stage in
# $WQ_DIR/queue/CURRENT_STAGE. Workers claim shards with an atomic `mv` from
# *_pending to *_running: the first `mv` to succeed owns the shard, and the
# others fail silently.
STAGE_FILE="$WQ_DIR/queue/CURRENT_STAGE"
IDLE_LIMIT=240   # exit after 240*5s = 20 min with no stage and no work
idle=0

while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "ERROR: vLLM died. Last 40 lines:"; tail -40 "$VLLM_LOG"; exit 1
    fi

    stage=""
    [ -f "$STAGE_FILE" ] && stage="$(cat "$STAGE_FILE" 2>/dev/null || true)"

    if [ "$stage" = "DONE" ]; then
        log "Orchestrator signalled DONE. Exiting."
        break
    fi
    if [ -z "$stage" ]; then
        idle=$((idle + 1))
        [ "$idle" -ge "$IDLE_LIMIT" ] && { log "Idle ${IDLE_LIMIT} cycles, no stage. Exiting."; break; }
        sleep 5; continue
    fi

    pending_dir="$WQ_DIR/queue/${stage}_pending"
    running_dir="$WQ_DIR/queue/${stage}_running"
    done_dir="$WQ_DIR/queue/${stage}_done"
    [ -d "$pending_dir" ] || { sleep 5; continue; }

    # Atomically claim one pending shard.
    claimed=""
    for f in "$pending_dir"/shard_*; do
        [ -e "$f" ] || continue
        name=$(basename "$f")
        if mv "$f" "$running_dir/$name" 2>/dev/null; then
            claimed="$name"; break
        fi
    done

    if [ -z "$claimed" ]; then
        # No pending shards for this stage: wait for the orchestrator to
        # advance the stage or for stragglers to finish.
        idle=$((idle + 1))
        [ "$idle" -ge "$IDLE_LIMIT" ] && { log "Idle ${IDLE_LIMIT} cycles. Exiting."; break; }
        sleep 5; continue
    fi
    idle=0
    CLAIMED_STAGE="$stage"
    CLAIMED_FILE="$running_dir/$claimed"

    log "[$stage] claimed $claimed (GPU=$GPU_NAME)"
    if process_shard "$stage" "$claimed"; then
        mv "$running_dir/$claimed" "$done_dir/$claimed"
        log "[$stage] $claimed DONE"
    else
        mv "$running_dir/$claimed" "$pending_dir/$claimed"
        log "[$stage] $claimed FAILED, re-queued"
        sleep 5
    fi
    CLAIMED_FILE=""; CLAIMED_STAGE=""
done

log "Worker finished."

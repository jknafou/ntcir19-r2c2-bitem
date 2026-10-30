#!/usr/bin/env bash
# NER work-queue DRIVER — faithful port of the production driver
# scripts/orchestration/workqueue_orchestrator.sh `do_go` (lines 154-820).
#
# Responsibilities (CPU, login node — submits SLURM, never computes):
#   1. Resolve paths + --demo down-scaling (via _demo_arg.sh).
#   2. Split $CHUNKS_DIR/passages.jsonl into NUM_SHARDS weight-balanced shards
#      (greedy round-robin by per-document passage count — identical algorithm
#      to production do_split, lines 68-149). Documents are never split across
#      shards (document-level extraction needs whole docs).
#   3. Write $WQ_DIR/config (consumed by both worker + orchestrator sbatch).
#   4. Populate the extract pending queue.
#   5. Submit the GPU worker array (ner_worker.sbatch) on the local cluster.
#   6. Submit the CPU orchestrator (ner_orchestrator.sbatch) that sequences the
#      stages and emits passages_enriched_v3.jsonl.
#
# DESIGN: work-queue, NOT a single job and NOT a 7-job afterok DAG. N shards are
# dispatched dynamically to W GPU workers; a fast GPU simply claims more shards.
# vLLM runs INSIDE each worker (localhost only) — no GPU<->CPU VLAN dependency.
#
# GPU VERSATILITY: workers request --gres=gpu:1 with a broad OR-constraint
# (a100_80gb_pcie | h100_nvl | h200_nvl | rtx_pro_6000_blackwell) so a 9B@TP=1
# replica lands on ANY capable card. SINGLE-CLUSTER: plain `sbatch` submits to
# whatever cluster you launch from — runs unchanged on bamboo or baobab. To use
# both clusters, launch this command from each; both drain the same shared queue.
#
# Model: Qwen/Qwen3.5-9B (NER model — NOT the 122B agent/judge; the standalone
# clients' Qwen3-32B-FP8 default is cosmetic and ignored by the single-model
# server, so we always pass --model explicitly). SAME model in --demo.
#
# ---------------------------------------------------------------------------
# DEMO vs PRODUCTION knobs (data/deployment volume only — never the algorithm,
# model, per-GPU TP, stage flags, --gres, or constraint):
#
#                 PRODUCTION (default)        DEMO (--demo)
#   shards        NER_SHARDS=100              NER_DEMO_SHARDS=32
#   gpu workers   NER_GPUS=20                 NER_DEMO_GPUS=8
#                 (shards:workers ~5:1 prod, 4:1 demo — oversubscribe so a fast
#                  GPU claims more shards and no card idles on the slow tail)
#   corpus        $CHUNKS_DIR (full)          data/r2c2_corpus_demo_N10000/chunks
#   model         Qwen/Qwen3.5-9B             Qwen/Qwen3.5-9B  (same)
#   per-GPU TP    1                           1                (same)
#   constraint    a100|h100|h200|blackwell    same             (same)
#
# All knobs are env-overridable:
#   NER_SHARDS / NER_GPUS               production shard + worker counts
#   NER_DEMO_SHARDS / NER_DEMO_GPUS     demo shard + worker counts
#   MODEL MAX_MODEL_LEN CROSS_DOC_IQR_K NER model + len + iqr-k
#
# Usage:
#   bash src/r2c2_index/slurm/ner_workqueue.sh            # production: 100 shards / 20 workers
#   bash src/r2c2_index/slurm/ner_workqueue.sh --demo     # demo:       32 shards /  8 workers
#   NER_GPUS=8 bash src/r2c2_index/slurm/ner_workqueue.sh # custom worker count

set -euo pipefail

DEMO=0
for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
        -h|--help) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "WARN: unknown arg: $arg" >&2 ;;
    esac
done

# _demo_arg.sh is the single source of DEMO_N and CHUNKS_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT="${PROJECT:-$REPO_ROOT}"
cd "$REPO_ROOT"
source "$REPO_ROOT/src/r2c2_index/slurm/_demo_arg.sh"

CHUNKS_DIR="${CHUNKS_DIR:-$REPO_ROOT/data/chunks}"
DOCUMENTS_JSONL="${DOCUMENTS_JSONL:-$CHUNKS_DIR/documents.jsonl}"
INPUT="${INPUT:-$CHUNKS_DIR/passages.jsonl}"
OUTPUT_FILE="$CHUNKS_DIR/passages_enriched_v3.jsonl"
ENTITY_MAP="$CHUNKS_DIR/entity_type_map.json"

LOGS_DIR="$REPO_ROOT/src/r2c2_index/slurm/logs"
mkdir -p "$LOGS_DIR" "$CHUNKS_DIR"

# One work-queue dir per corpus, so demo and production never collide.
WQ_DIR="${WQ_DIR:-$CHUNKS_DIR/.ner_workqueue}"

MODEL_ID="${MODEL:-Qwen/Qwen3.5-9B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
CROSS_DOC_IQR_K="${CROSS_DOC_IQR_K:-1.5}"

if [ "$DEMO" = 1 ]; then
    NUM_SHARDS="${NER_DEMO_SHARDS:-32}"
    NUM_GPUS="${NER_DEMO_GPUS:-8}"
else
    NUM_SHARDS="${NER_SHARDS:-100}"
    NUM_GPUS="${NER_GPUS:-20}"
fi

# Python interpreter, which must be able to run vLLM. `uv sync` builds the one
# project venv (Python 3.12, the vLLM + CUDA stack, every stage's deps) and the
# workers serve the model from it. VLLM_PYTHON optionally points at a different
# interpreter; the default is the project `.venv`. Workers and orchestrator read
# PYTHON from $WQ_DIR/config, so it is resolved once here.
if [ -n "${VLLM_PYTHON:-}" ] && [ -x "$VLLM_PYTHON" ]; then
    PYTHON="$VLLM_PYTHON"
elif [ -x "$REPO_ROOT/.venv/bin/python3" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python3"
else
    PYTHON=python3
fi

log() { echo "[$(date '+%H:%M:%S')] [ner-wq] $*"; }

[ -f "$INPUT" ] || { log "ERROR: input not found: $INPUT (run Phase 1 / 01_chunk.sh first)"; exit 1; }
[ -f "$DOCUMENTS_JSONL" ] || { log "ERROR: documents not found: $DOCUMENTS_JSONL"; exit 1; }

echo "=== NER work-queue $([ "$DEMO" = 1 ] && echo "(--demo)") ==="
log "  repo:       $REPO_ROOT"
log "  input:      $INPUT"
log "  documents:  $DOCUMENTS_JSONL"
log "  output:     $OUTPUT_FILE"
log "  model:      $MODEL_ID (TP=1 per GPU)"
log "  shards:     $NUM_SHARDS    gpu workers: $NUM_GPUS"
log "  wq_dir:     $WQ_DIR"

mkdir -p "$WQ_DIR"/{shards,jobs,endpoints,output/{logs,shards}}
mkdir -p "$WQ_DIR"/queue/{extract_pending,extract_running,extract_done}
mkdir -p "$WQ_DIR"/queue/{refine_pending,refine_running,refine_done}
mkdir -p "$WQ_DIR"/queue/{crossdoc_review_pending,crossdoc_review_running,crossdoc_review_done}

# Split into weight-balanced shards (greedy round-robin by passage count).
# A prior split is reused only if its shard count matches.
need_split=1
if [ -f "$WQ_DIR/split_meta.json" ]; then
    existing=$("$PYTHON" -c "import json;print(json.load(open('$WQ_DIR/split_meta.json'))['num_shards'])" 2>/dev/null || echo "")
    if [ "$existing" = "$NUM_SHARDS" ]; then
        log "split already exists ($NUM_SHARDS shards), reusing"
        need_split=0
    fi
fi

if [ "$need_split" = 1 ]; then
    log "Splitting $INPUT into $NUM_SHARDS weight-balanced shards..."
    "$PYTHON" - "$INPUT" "$NUM_SHARDS" "$WQ_DIR" <<'PYSPLIT'
import json, sys, os
from collections import Counter

input_path, num_shards, wq_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# Pass 1: per-document passage counts (preserve first-seen order).
order, seen, counts = [], set(), {}
with open(input_path) as f:
    for line in f:
        did = json.loads(line)["doc_id"]
        if did not in seen:
            seen.add(did); order.append(did); counts[did] = 0
        counts[did] += 1
num_docs = len(order); total = sum(counts.values())
print(f"  {num_docs:,} documents, {total:,} passages")

# Greedy round-robin: assign heaviest docs first to the lightest shard.
loads = [0] * num_shards
assign = {}
for did in sorted(order, key=lambda d: counts[d], reverse=True):
    i = min(range(num_shards), key=lambda k: loads[k])
    assign[did] = i; loads[i] += counts[did]

shard_dir = os.path.join(wq_dir, "shards")
files = [open(os.path.join(shard_dir, f"shard_{i:04d}.jsonl"), "w") for i in range(num_shards)]
written = [0] * num_shards
with open(input_path) as f:
    for line in f:
        i = assign[json.loads(line)["doc_id"]]
        files[i].write(line); written[i] += 1
for fh in files:
    fh.close()

dc = Counter(assign.values())
meta = {
    "input": os.path.abspath(input_path),
    "num_shards": num_shards,
    "num_docs": num_docs,
    "total_passages": total,
    "shard_docs": [dc.get(i, 0) for i in range(num_shards)],
    "shard_passages": written,
}
with open(os.path.join(wq_dir, "split_meta.json"), "w") as f:
    json.dump(meta, f, indent=2); f.write("\n")
lo, hi = min(loads), max(loads)
print(f"  balance: min={lo:,} max={hi:,} ratio={hi/max(1,lo):.3f}")
PYSPLIT
    log "split complete"
fi

# Populate the extract queue, skipping shards already done on resume.
for f in "$WQ_DIR"/shards/shard_*.jsonl; do
    [ -e "$f" ] || continue
    sname=$(basename "$f" .jsonl)
    [ -f "$WQ_DIR/queue/extract_done/$sname" ] || touch "$WQ_DIR/queue/extract_pending/$sname"
done
log "extract queue: $(find "$WQ_DIR/queue/extract_pending" -name 'shard_*' | wc -l) pending"

# Resolve the optional prompt dir (best -> iter_0 -> base).
PROMPT_DIR=""
MODEL_NAME="qwen3.5-9b"
for cand in "$SCRIPT_DIR/prompts/$MODEL_NAME/best" "$SCRIPT_DIR/prompts/$MODEL_NAME/iter_0" "$SCRIPT_DIR/prompts/base"; do
    [ -d "$cand" ] && { PROMPT_DIR="$cand"; break; }
done

# $WQ_DIR/config is the single source of truth for both sbatch scripts.
cat > "$WQ_DIR/config" <<CONF
REPO_ROOT="$REPO_ROOT"
PROJECT="$PROJECT"
WQ_DIR="$WQ_DIR"
PYTHON="$PYTHON"
INPUT="$INPUT"
DOCUMENTS_JSONL="$DOCUMENTS_JSONL"
OUTPUT_FILE="$OUTPUT_FILE"
ENTITY_MAP="$ENTITY_MAP"
MODEL_ID="$MODEL_ID"
MAX_LEN="$MAX_MODEL_LEN"
CROSS_DOC_IQR_K="$CROSS_DOC_IQR_K"
NUM_SHARDS=$NUM_SHARDS
PROMPT_DIR="$PROMPT_DIR"
CONF
log "wrote $WQ_DIR/config"

# Clear any stale stage marker so newly submitted workers idle until the
# orchestrator activates extract.
: > "$WQ_DIR/queue/CURRENT_STAGE"

# Submit the worker array and the orchestrator on the local cluster only.
#
# Plain `sbatch` (no `-M`) submits to whatever cluster this is launched from, so
# the same command works on bamboo or baobab. Cross-cluster `-M` fan-out is
# avoided because it depends on SLURM federation and, when that fails, silently
# halves the worker count. To use both clusters for a big run, launch this same
# command from each: both drain the same $WQ_DIR queue on shared BeeGFS, so
# workers on either side claim shards atomically. All NUM_GPUS workers land
# locally.
WORKER_SBATCH="$REPO_ROOT/src/r2c2_index/slurm/ner_worker.sbatch"
ORCH_SBATCH="$REPO_ROOT/src/r2c2_index/slurm/ner_orchestrator.sbatch"

local_max=$((NUM_GPUS - 1))
log "submitting worker array (local cluster): 0-$local_max ($NUM_GPUS GPUs)"
LOCAL_JID=$(WQ_DIR="$WQ_DIR" sbatch --parsable \
    --array=0-"$local_max" \
    --export=ALL,WQ_DIR="$WQ_DIR" \
    --chdir="$REPO_ROOT" \
    "$WORKER_SBATCH")
echo "local $LOCAL_JID" > "$WQ_DIR/jobs/worker_local.jobid"
log "  worker array: $LOCAL_JID"

log "submitting orchestrator (local cluster)"
ORCH_JID=$(WQ_DIR="$WQ_DIR" sbatch --parsable \
    --export=ALL,WQ_DIR="$WQ_DIR" \
    --chdir="$REPO_ROOT" \
    "$ORCH_SBATCH")
echo "local $ORCH_JID" > "$WQ_DIR/jobs/orchestrator.jobid"
log "  orchestrator: $ORCH_JID"

echo ""
echo "============================================"
echo "  NER WORK-QUEUE SUBMITTED (fire and forget)"
echo "============================================"
echo "  shards:        $NUM_SHARDS"
echo "  workers:       $NUM_GPUS (single cluster)"
echo "  worker array:  $LOCAL_JID"
echo "  orchestrator:  $ORCH_JID"
echo "  final artifact: $OUTPUT_FILE"
echo ""
echo "  Monitor:"
echo "    tail -f $LOGS_DIR/ner_wq_orch_${ORCH_JID}.out"
echo "    squeue -j $ORCH_JID,$LOCAL_JID"
echo "  Cancel everything:"
echo "    scancel $ORCH_JID $LOCAL_JID"
echo "============================================"

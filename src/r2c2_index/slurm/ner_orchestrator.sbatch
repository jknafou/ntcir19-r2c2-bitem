#!/usr/bin/env bash
# NER work-queue orchestrator (CPU job, no GPU).
#
# It owns stage ordering, queue population, the merge between stages, the
# single CPU cross-doc precompute, and the final convert + offset merge that
# emits passages_enriched_v3.jsonl.
#
# It does not spawn workers over the network. It publishes the active
# stage to $WQ_DIR/queue/CURRENT_STAGE; the GPU array workers
# (ner_worker.sbatch) drain that stage's pending queue against their own local
# vLLM. The orchestrator polls the *_done counters and advances when a stage's
# shards are all done. When everything is finished it writes "DONE" so workers
# self-terminate, then scancels the worker array(s).
#
# Three GPU stages plus one CPU precompute, each over all shards:
#   Stage 1  extract           : all shards -> shard_*/extraction.jsonl  -> merge
#   Stage 2  refine (within)    : all shards -> shard_*/refined.jsonl     -> merge
#   Stage 3a fast_crossdoc      : once over the shards dir (CPU, no LLM)  -> crossdoc_candidates.json
#   Stage 3b crossdoc_review    : all shards -> shard_*/refined_crossdoc.jsonl -> merge
#   Finalize convert_ner_to_enriched + merge_offsets_to_enriched -> passages_enriched_v3.jsonl
#
#SBATCH --job-name=r2c2-ner-wq-orch
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=src/r2c2_index/slurm/logs/ner_wq_orch_%j.out
#SBATCH --error=src/r2c2_index/slurm/logs/ner_wq_orch_%j.err

set -euo pipefail

WQ_DIR="${WQ_DIR:?WQ_DIR must be set (exported by ner_workqueue.sh)}"
source "$WQ_DIR/config"

log() { echo "[$(date '+%H:%M:%S')] [ner-orch] $*"; }
run() { PYTHONPATH="$REPO_ROOT/src" PYTHONUNBUFFERED=1 "$PYTHON" -m "$@"; }

STAGE_FILE="$WQ_DIR/queue/CURRENT_STAGE"

trap 'log "USR1 — walltime. Leaving queues intact for resume."; exit 0' USR1

cleanup() {
    # On any exit, requeue running shards of the active stage so a resume can
    # reclaim them, and clear CURRENT_STAGE so a fresh worker array idles
    # instead of churning the wrong stage.
    local st; st="$(cat "$STAGE_FILE" 2>/dev/null || true)"
    if [ -n "$st" ] && [ "$st" != "DONE" ]; then
        for f in "$WQ_DIR/queue/${st}_running"/shard_*; do
            [ -e "$f" ] && mv "$f" "$WQ_DIR/queue/${st}_pending/$(basename "$f")" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT

# populate_queue: touch a pending marker for every shard that has the required
# input and is not already done. Idempotent, so safe on resume.
#   $1 stage; $2 required-input filename inside shard dir ("" = source shard)
populate_queue() {
    local stage="$1" need="$2"
    mkdir -p "$WQ_DIR/queue/${stage}_pending" \
             "$WQ_DIR/queue/${stage}_running" \
             "$WQ_DIR/queue/${stage}_done"
    local i sname
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        sname=$(printf "shard_%04d" "$i")
        [ -f "$WQ_DIR/queue/${stage}_done/$sname" ] && continue
        if [ -z "$need" ] || [ -f "$WQ_DIR/output/shards/$sname/$need" ]; then
            touch "$WQ_DIR/queue/${stage}_pending/$sname"
        fi
    done
}

# wait_stage: set CURRENT_STAGE and block until all shards reach *_done.
# Workers do the claiming and processing; this only sequences and reports.
wait_stage() {
    local stage="$1"
    local pending_dir="$WQ_DIR/queue/${stage}_pending"
    local running_dir="$WQ_DIR/queue/${stage}_running"
    local done_dir="$WQ_DIR/queue/${stage}_done"

    echo "$stage" > "$STAGE_FILE"
    log "=== STAGE ${stage^^} active ==="

    local stall=0
    while true; do
        local p r d
        p=$(find "$pending_dir" -maxdepth 1 -name 'shard_*' 2>/dev/null | wc -l)
        r=$(find "$running_dir" -maxdepth 1 -name 'shard_*' 2>/dev/null | wc -l)
        d=$(find "$done_dir"    -maxdepth 1 -name 'shard_*' 2>/dev/null | wc -l)

        if [ "$p" -eq 0 ] && [ "$r" -eq 0 ]; then
            log "$stage complete: $d/$NUM_SHARDS shards done"
            break
        fi
        log "$stage: $d done, $r running, $p pending (of $NUM_SHARDS)"

        # Warn when no worker endpoint answers for a long time, but keep
        # waiting: the workers may still be queued or rescheduling.
        local healthy=0
        for ef in "$WQ_DIR/endpoints"/worker_*.endpoint; do
            [ -f "$ef" ] || continue
            curl -sf "$(cat "$ef" | sed 's#/v1$#/v1/models#')" --connect-timeout 5 >/dev/null 2>&1 && healthy=$((healthy + 1))
        done
        if [ "$healthy" -eq 0 ]; then
            stall=$((stall + 1))
            [ $((stall % 10)) -eq 0 ] && log "WARN: no healthy worker endpoints (waiting for GPU array to schedule)"
        else
            stall=0
        fi
        sleep 30
    done
}

# merge_stage: cat per-shard outputs into one merged file.
#   $1 filename inside shard dir; $2 merged output basename
merge_stage() {
    local fname="$1" out="$WQ_DIR/output/$2"
    log "Merging $fname from $NUM_SHARDS shards -> $out"
    : > "$out"
    local i sname f total=0 c
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        sname=$(printf "shard_%04d" "$i")
        f="$WQ_DIR/output/shards/$sname/$fname"
        if [ -f "$f" ]; then
            c=$(wc -l < "$f"); total=$((total + c)); cat "$f" >> "$out"
        fi
    done
    log "  merged $total passages -> $out"
}

mkdir -p "$WQ_DIR/output"/{logs,shards}

log "Work-queue orchestrator starting"
log "  shards=$NUM_SHARDS  model=$MODEL_ID  output=$OUTPUT_FILE"

# Stage 1: extract (all shards)
populate_queue extract ""
wait_stage extract
merge_stage extraction.jsonl extraction.jsonl

# Stage 2: refine within-doc (all shards)
populate_queue refine extraction.jsonl
wait_stage refine
merge_stage refined.jsonl refined.jsonl

# Stage 3a: cross-doc candidate precompute (CPU, once, no LLM)
# Clear the stage so workers idle while this CPU job runs.
echo "" > "$STAGE_FILE"
CROSSDOC_CANDIDATES="$WQ_DIR/output/crossdoc_candidates.json"
if [ -f "$CROSSDOC_CANDIDATES" ]; then
    log "cross-doc candidates exist, skipping fast_crossdoc"
else
    log "=== STAGE CROSSDOC-CANDIDATES (CPU, workers=8, iqr-k=$CROSS_DOC_IQR_K) ==="
    run r2c2_index.ner fast_crossdoc \
        --corpus "$WQ_DIR/output/shards" \
        --documents "$DOCUMENTS_JSONL" \
        --output "$CROSSDOC_CANDIDATES" \
        --workers 8 \
        --iqr-k "$CROSS_DOC_IQR_K" \
        --tmpdir "/tmp/fast_crossdoc_${SLURM_JOB_ID:-$$}" \
        2>&1 | tee "$WQ_DIR/output/logs/crossdoc_candidates.log"
    [ -f "$CROSSDOC_CANDIDATES" ] || { log "ERROR: fast_crossdoc produced no candidates"; exit 1; }
fi
log "candidates: $(stat -c%s "$CROSSDOC_CANDIDATES" 2>/dev/null || echo 0) bytes"

# Stage 3b: cross-doc LLM review (all shards)
populate_queue crossdoc_review refined.jsonl
wait_stage crossdoc_review
merge_stage refined_crossdoc.jsonl refined_crossdoc.jsonl

# Finalize: signal workers to exit, then convert and merge offsets
echo "DONE" > "$STAGE_FILE"
log "All GPU stages complete; signalled workers DONE."

# Cancel the worker array(s) so the GPUs are freed immediately.
for jf in "$WQ_DIR/jobs"/worker_*.jobid; do
    [ -f "$jf" ] || continue
    jline="$(cat "$jf")"               # format: "<cluster> <jobid>"
    jcluster="${jline%% *}"; jid="${jline##* }"
    log "scancel worker array $jid on $jcluster"
    if [ "$jcluster" = "local" ]; then
        scancel "$jid" 2>/dev/null || true
    else
        scancel -M "$jcluster" "$jid" 2>/dev/null || true
    fi
done

ENRICHED="$WQ_DIR/output/passages_enriched.jsonl"

log "--- convert_ner_to_enriched ---"
run r2c2_index.ner convert_ner_to_enriched \
    --input "$WQ_DIR/output/refined_crossdoc.jsonl" \
    --output "$ENRICHED" \
    --stats \
    2>&1 | tee "$WQ_DIR/output/logs/convert_enriched.log"
log "  enriched: $(wc -l < "$ENRICHED") passages"

# Single-process, full-corpus offset merge -> passages_enriched_v3.jsonl
log "--- merge_offsets_to_enriched -> passages_enriched_v3 ---"
run r2c2_index.ner merge_offsets_to_enriched \
    --passages-v3 "$INPUT" \
    --enriched "$ENRICHED" \
    --output "$OUTPUT_FILE" \
    --entity-map "$ENTITY_MAP" \
    2>&1 | tee "$WQ_DIR/output/logs/merge_offsets.log"

[ -s "$OUTPUT_FILE" ] || { log "ERROR: $OUTPUT_FILE missing/empty"; exit 1; }
log "=== NER PIPELINE COMPLETE ==="
log "  extraction: $(wc -l < "$WQ_DIR/output/extraction.jsonl") passages"
log "  refined:    $(wc -l < "$WQ_DIR/output/refined.jsonl") passages"
log "  cross-doc:  $(wc -l < "$WQ_DIR/output/refined_crossdoc.jsonl") passages"
log "  FINAL:      $OUTPUT_FILE ($(wc -l < "$OUTPUT_FILE") passages)"

trap - EXIT

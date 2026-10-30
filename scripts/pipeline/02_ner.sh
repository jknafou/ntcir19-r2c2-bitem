#!/usr/bin/env bash
# Phase 2 — NER pipeline (WORK-QUEUE design).
#
# Reads $CHUNKS_DIR/passages.jsonl (output of Phase 1) and produces
# $CHUNKS_DIR/passages_enriched_v3.jsonl (entity-annotated passages consumed by
# Phase 3 KG, Phase 4 ES, Phase 5 FAISS).
#
# WORK-QUEUE DESIGN (faithful port of the production driver
# scripts/orchestration/workqueue_orchestrator.sh — NOT a single job, NOT a
# 7-job afterok DAG):
#
#   shard split -> dispatch -> GPU workers -> stage flow -> merge -> enriched
#
#   1. SPLIT   passages.jsonl into N weight-balanced shards (greedy round-robin
#              by per-document passage count; documents never split across
#              shards). Production: 100 shards.
#   2. DISPATCH a CPU orchestrator (ner_orchestrator.sbatch) sequences the
#              stages by publishing the active stage to a shared queue dir;
#              W GPU workers (ner_worker.sbatch, a SLURM array) each run vLLM on
#              localhost and atomically CLAIM the next pending shard via a
#              file-rename lock (mv pending->running), process it, then mark it
#              done (mv running->done). Fast GPUs claim more shards. No idle GPU.
#   3. STAGE FLOW over ALL shards:
#        extract          (extract_entities_llm --document-level --no-propagate)
#        refine within-doc (refine_entities)
#        fast_crossdoc     (ONCE, CPU, no LLM -> crossdoc_candidates.json)
#        crossdoc_review   (refine_entities --cross-document --precomputed-crossdoc)
#   4. MERGE   per-shard outputs are `cat`-merged after each GPU stage, then
#              convert_ner_to_enriched + merge_offsets_to_enriched produce the
#              final passages_enriched_v3.jsonl.
#
# vLLM runs INSIDE each worker (localhost) — no GPU<->CPU VLAN dependency.
# Workers are --gres=gpu:1 with a broad OR-constraint
# (a100_80gb_pcie|h100_nvl|h200_nvl|rtx_pro_6000_blackwell) so a 9B@TP=1 replica
# lands on ANY capable card. SINGLE-CLUSTER: runs on whichever cluster you launch
# from (plain sbatch, no -M). Launch from each cluster to use both (shared queue).
#
# Production NER model: Qwen/Qwen3.5-9B (dense, TP=1) — NOT the 122B agent/judge,
# NOT the cosmetic Qwen3-32B-FP8 client default. SAME model in --demo.
#
# DEMO vs PRODUCTION knobs (deployment volume only — same model/hardware-type):
#                  PRODUCTION (default)   DEMO (--demo)
#   shards         100                    4         (override: NER_DEMO_SHARDS)
#   gpu workers    20                     2         (override: NER_DEMO_GPUS)
#   corpus         data/chunks            data/r2c2_corpus_demo_N10000/chunks
#   model          Qwen/Qwen3.5-9B        Qwen/Qwen3.5-9B   (same)
#
# Run after Phase 1 (01_chunk.sh) has finished.
#
# Usage:
#   bash scripts/pipeline/02_ner.sh            # production: 100 shards / 20 workers
#   bash scripts/pipeline/02_ner.sh --demo     # demo:       32 shards /  8 workers

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

echo "=== Phase 2: NER work-queue $([ "$DEMO" = 1 ] && echo "(--demo)") ==="
echo "  Repo: $REPO_ROOT"
if [ "$DEMO" = 1 ]; then
    echo "  Demo deployment (work-queue) — resolved shard/worker counts printed below"
    echo "  (override with NER_DEMO_SHARDS / NER_DEMO_GPUS; defaults 32 / 8)"
else
    echo "  Production deployment (work-queue) — resolved shard/worker counts printed below"
    echo "  (override with NER_SHARDS / NER_GPUS; defaults 100 / 20)"
fi
echo ""

# Delegate to the work-queue driver. It splits the corpus, submits the GPU
# worker array (multi-cluster via sbatch -M when the secondary is reachable),
# and submits the CPU orchestrator that sequences the stages and emits
# passages_enriched_v3.jsonl. The driver parses --demo itself (via _demo_arg.sh)
# to remap CHUNKS_DIR + scale shard/worker counts; the model is unchanged.
exec bash "$REPO_ROOT/src/r2c2_index/slurm/ner_workqueue.sh" $DEMO_ARG

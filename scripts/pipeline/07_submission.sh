#!/usr/bin/env bash
# Phase 7 — Submission-ready bundle (single command).
#
# Re-aggregates the agent run's trace bundle under each of the four AC
# strategies, then packages the results into an upload-ready directory
# whose layout mirrors submission/ac-2026-05-15/:
#
#     $OUTDIR/
#     ├── BITEM-PG-1            (un-zipped passages of BITEM-PR.zip)
#     ├── BITEM-PG-2
#     ├── BITEM-PR.zip          (copied from Phase 5 — the PR is already final)
#     ├── BITEM-AC-1            (trace-redesign)
#     ├── BITEM-AC-2            (max-conf)
#     ├── BITEM-AC-3            (conf-sum-cluster)
#     ├── BITEM-AC-4            (llm-judge)
#     ├── BITEM-AC.zip          (the 4 AC files zipped at root)
#     └── submission.SHA256
#
# So the answer to "does this produce both PR and AC?" is YES — the
# PR comes from Phase 5 unchanged (we copy, not regenerate); the AC
# is freshly built from the four strategies.
#
# Submitted SLURM jobs (all chained via afterok / afterany):
#   1-3.  ac-trace-redesign / ac-max-conf / ac-conf-sum-cluster   (CPU, parallel)
#   4.    llm_judge.sbatch                                        (GPU; vLLM for AC-4)
#   5.    ac4-judge      (CPU; waits on llm_judge endpoint file)
#   6.    ac4-reaggregate (CPU; afterok:5)
#   7.    cleanup        (CPU; afterany:6 — scancels job 4)
#   8.    bundle         (CPU; afterok:1,2,3,6 — assembles the upload-ready
#                         dir above and computes submission.SHA256)
#
# Auto-discovers TRACES (most recent data/runs/pr_* in production, or
# most recent submission/demo-*/ in --demo). PR_DIR / QUESTIONS pick
# the matching variant. Override any of TRACES / PR_DIR / QUESTIONS /
# OUTDIR via env vars before invoking.
#
# Usage:
#   bash scripts/pipeline/07_submission.sh
#   bash scripts/pipeline/07_submission.sh --demo

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

TS="$(date +%Y%m%d_%H%M%S)"
if [ "$DEMO" = 1 ]; then
    TRACES="${TRACES:-$(ls -dt "$REPO_ROOT"/submission/demo-* 2>/dev/null | head -1 || true)}"
    QUESTIONS="${QUESTIONS:-$REPO_ROOT/data/r2c2/dry_run_questions.jsonl}"
    PR_DIR="${PR_DIR:-${TRACES%/}/submission}"
    OUTDIR="${OUTDIR:-$REPO_ROOT/submission/produced-${TS}_demo}"
else
    TRACES="${TRACES:-$(ls -dt "$REPO_ROOT"/data/runs/pr_* 2>/dev/null | head -1 || true)}"
    QUESTIONS="${QUESTIONS:-$REPO_ROOT/data/r2c2/test_questions.jsonl}"
    PR_DIR="${PR_DIR:-$REPO_ROOT/submission/pr-2026-04-25}"
    OUTDIR="${OUTDIR:-$REPO_ROOT/submission/produced-${TS}}"
fi

if [ -z "$TRACES" ] || [ ! -d "$TRACES" ]; then
    echo "ERROR: TRACES dir not found. Run Phase 5 (06_trace.sh) first," >&2
    echo "       or set TRACES=/path/to/run explicitly." >&2
    exit 2
fi
mkdir -p "$OUTDIR"

echo "=== Phase 6: post-hoc AC strategy panel $([ "$DEMO" = 1 ] && echo "(--demo)") ==="
echo "  TRACES:    $TRACES"
echo "  PR_DIR:    $PR_DIR"
echo "  QUESTIONS: $QUESTIONS"
echo "  OUTDIR:    $OUTDIR"
echo ""

# 1-3: three closed-form CPU re-aggregations (parallel, no deps)
JIDS=()
for strategy in trace-redesign max-conf conf-sum-cluster; do
    SUB="$OUTDIR/$strategy"; mkdir -p "$SUB"
    JID=$(sbatch --parsable \
        --job-name=ac-$strategy --partition=shared-cpu \
        --cpus-per-task=2 --mem=4G --time=00:15:00 \
        --output="$SUB/run.out" \
        --wrap="cd $REPO_ROOT && PYTHONPATH=$REPO_ROOT/src ./.venv/bin/python -m r2c2_agent reaggregate-ac \
            --strategy '$strategy' \
            --pr-dir '$PR_DIR' \
            --traces-dir '$TRACES' \
            --source-questions '$QUESTIONS' \
            --output '$SUB'")
    echo "  [1-3/7] ac-$strategy job=$JID"
    JIDS+=("$JID")
done

# 4: vLLM endpoint for AC-4 (--demo same model, same hardware)
JUDGE_VLLM_JID=$(sbatch --parsable src/r2c2_agent/slurm/llm_judge.sbatch $DEMO_ARG)
echo "  [4/7]   llm-judge vLLM job=$JUDGE_VLLM_JID (~10-15 min boot)"

# 5: AC-4 judge pass. It needs the vLLM endpoint, not the CPU
# re-aggregations, so the wrapped command polls for the endpoint file
# and the job can be queued immediately.
SUB="$OUTDIR/llm-judge"; mkdir -p "$SUB"
JUDGE_WRAP=$(cat <<EOF
cd "$REPO_ROOT" \
&& until [ -f src/r2c2_agent/slurm/endpoints/llm_judge_${JUDGE_VLLM_JID}.endpoint ]; do sleep 10; done \
&& VLLM_ENDPOINT=\$(jq -r .url src/r2c2_agent/slurm/endpoints/llm_judge_${JUDGE_VLLM_JID}.endpoint) \
&& echo "judge endpoint: \$VLLM_ENDPOINT" \
&& PYTHONPATH=$REPO_ROOT/src ./.venv/bin/python -m r2c2_agent llm-judge \
       --vllm-endpoint "\$VLLM_ENDPOINT" \
       --model Qwen/Qwen3.5-122B-A10B-FP8 \
       --pr-dir '$PR_DIR' \
       --traces-dir '$TRACES' \
       --questions '$QUESTIONS' \
       --output '$SUB/judge_decisions.jsonl'
EOF
)
JUDGE_JID=$(sbatch --parsable \
    --job-name=ac4-judge --partition=shared-cpu \
    --cpus-per-task=2 --mem=4G --time=01:00:00 \
    --output="$SUB/judge.out" \
    --wrap="$JUDGE_WRAP")
echo "  [5/7]   ac4-judge      job=$JUDGE_JID (waits on $JUDGE_VLLM_JID's endpoint file)"

# 6: AC-4 reaggregate — depends on judge pass
REAGG_JID=$(sbatch --parsable \
    --dependency=afterok:$JUDGE_JID \
    --job-name=ac4-reagg --partition=shared-cpu \
    --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="$SUB/reagg.out" \
    --wrap="cd $REPO_ROOT && PYTHONPATH=$REPO_ROOT/src ./.venv/bin/python -m r2c2_agent reaggregate-ac \
        --strategy llm-judge \
        --judge-decisions '$SUB/judge_decisions.jsonl' \
        --pr-dir '$PR_DIR' \
        --traces-dir '$TRACES' \
        --source-questions '$QUESTIONS' \
        --output '$SUB'")
echo "  [6/7]   ac4-reagg      job=$REAGG_JID (afterok:$JUDGE_JID)"

# 7: scancel the vLLM endpoint after reagg finishes
CLEANUP_JID=$(sbatch --parsable --kill-on-invalid-dep=yes \
    --job-name=ac4-cleanup --partition=shared-cpu \
    --cpus-per-task=1 --mem=1G --time=00:05:00 \
    --dependency=afterany:$REAGG_JID \
    --output="$SUB/cleanup.out" \
    --wrap="scancel $JUDGE_VLLM_JID")
echo "  [7/8]   cleanup        job=$CLEANUP_JID (afterany:$REAGG_JID — scancels $JUDGE_VLLM_JID)"

# 8: bundle — depends on every reaggregate (1-3 + AC-4's reagg) succeeding.
# Assembles the upload-ready directory layout and computes
# submission.SHA256. PR.zip is copied from Phase 5's output, not
# regenerated (the PR is finished as soon as Phase 5 ends).
BUNDLE_DEPS="$(IFS=:; echo "${JIDS[*]}"):$REAGG_JID"
BUNDLE_WRAP=$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"

# Locate Phase 5's PR.zip + unzipped PG files. orchestrator.sbatch
# writes them to <trace_dir>/submission/.
PHASE5_SUB="${TRACES%/}/submission"
if [ ! -f "\$PHASE5_SUB/BITEM-PR.zip" ]; then
    echo "ERROR: Phase 5 PR.zip not found at \$PHASE5_SUB/BITEM-PR.zip" >&2
    echo "       Did Phase 5 (06_trace.sh) finish successfully?"     >&2
    exit 2
fi

# Copy PR side of the submission.
cp "\$PHASE5_SUB/BITEM-PR.zip" "$OUTDIR/"
[ -f "\$PHASE5_SUB/BITEM-PG-1" ] && cp "\$PHASE5_SUB/BITEM-PG-1" "$OUTDIR/"
[ -f "\$PHASE5_SUB/BITEM-PG-2" ] && cp "\$PHASE5_SUB/BITEM-PG-2" "$OUTDIR/"

# Collect AC-1..AC-4 from the four strategy sub-dirs into the bundle.
# reaggregate-ac writes BITEM-AC-<N>.v2 for trace-redesign (PG-1 path)
# and BITEM-AC-<N> for the others. We rename .v2 -> N so the bundle is
# uniform.
for pair in 1:trace-redesign 2:max-conf 3:conf-sum-cluster 4:llm-judge; do
    n="\${pair%%:*}"; strategy="\${pair##*:}"
    src=""
    for cand in "$OUTDIR/\$strategy/BITEM-AC-\$n" "$OUTDIR/\$strategy/BITEM-AC-\$n.v2"; do
        [ -f "\$cand" ] && src="\$cand" && break
    done
    if [ -z "\$src" ]; then
        echo "WARN: BITEM-AC-\$n not found under $OUTDIR/\$strategy/" >&2
        continue
    fi
    cp "\$src" "$OUTDIR/BITEM-AC-\$n"
done

# Build BITEM-AC.zip at zip root (matches the shipped layout).
cd "$OUTDIR"
zip -q BITEM-AC.zip BITEM-AC-1 BITEM-AC-2 BITEM-AC-3 BITEM-AC-4

# Manifest.
sha256sum BITEM-AC-1 BITEM-AC-2 BITEM-AC-3 BITEM-AC-4 BITEM-AC.zip \
          \$([ -f BITEM-PG-1 ] && echo BITEM-PG-1) \
          \$([ -f BITEM-PG-2 ] && echo BITEM-PG-2) \
          BITEM-PR.zip \
    > submission.SHA256
echo "=== bundled $OUTDIR ==="
ls -la "$OUTDIR"
echo "--- submission.SHA256 ---"
cat "$OUTDIR/submission.SHA256"
EOF
)
BUNDLE_JID=$(sbatch --parsable \
    --job-name=bundle-submission --partition=shared-cpu \
    --cpus-per-task=1 --mem=1G --time=00:05:00 \
    --dependency=afterok:$BUNDLE_DEPS \
    --output="$OUTDIR/bundle.out" \
    --wrap="$BUNDLE_WRAP")
echo "  [8/8]   bundle         job=$BUNDLE_JID (afterok:$BUNDLE_DEPS)"

echo ""
echo "All jobs submitted. After the bundle job finishes, the upload-ready"
echo "submission lives at:"
echo "  $OUTDIR/{BITEM-PR.zip, BITEM-AC.zip, BITEM-AC-1..4, BITEM-PG-{1,2}, submission.SHA256}"

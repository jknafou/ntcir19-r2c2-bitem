# Shared helper: parses a `--demo` positional argument from every
# r2c2_index sbatch (and src/r2c2_agent/slurm/llm_judge.sbatch) and remaps
# the canonical env vars to demo-sized locations.
#
# Source this from a sbatch script body after `PROJECT` is set:
#
#     PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
#     source "$PROJECT/src/r2c2_index/slurm/_demo_arg.sh"
#
# When the parent script is invoked as `sbatch <script> --demo`, this
# helper sets:
#   DEMO=1
#   DEMO_N        = 10000 (override with DEMO_N=N env var)
#   DEMO_ROOT     = $PROJECT/data/r2c2_corpus_demo_N${DEMO_N}
#   CHUNKS_DIR    = $DEMO_ROOT/chunks         (if caller did not set)
#   INDEXES_DIR   = $DEMO_ROOT/indexes        (if caller did not set)
#   ES_DATA_DIR   = $DEMO_ROOT/es-data        (if caller did not set)
#
# Stages whose work depends on the chunker's --limit flag (Stage 1 alone)
# can use `$DEMO_N` directly in their python invocation, e.g.:
#   [ "$DEMO" = 1 ] && CHUNK_ARGS="--limit $DEMO_N"
#
# Without `--demo`, DEMO=0 and no env vars are touched.

DEMO=0
for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
    esac
done

if [ "$DEMO" = 1 ]; then
    # Self-default PROJECT so this helper works even when sourced from a
    # `sbatch --wrap` body that never set it (e.g. the ner-extract job).
    # Callers that `cd "$SLURM_SUBMIT_DIR"` first land at the repo root,
    # so SLURM_SUBMIT_DIR / $PWD both resolve to $PROJECT correctly.
    PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
    DEMO_N="${DEMO_N:-10000}"
    # DEMO_N docs are reservoir-sampled uniformly from each corpus, so
    # the sub-corpus can cover the test questions; taking the first N
    # would cluster on archive order. The default seed of 42 keeps demo
    # runs reproducible; set DEMO_SEED=<int> for a different sample.
    DEMO_SEED="${DEMO_SEED:-42}"
    DEMO_ROOT="${DEMO_ROOT:-$PROJECT/data/r2c2_corpus_demo_N${DEMO_N}}"
    export CHUNKS_DIR="${CHUNKS_DIR:-$DEMO_ROOT/chunks}"
    export INDEXES_DIR="${INDEXES_DIR:-$DEMO_ROOT/indexes}"
    export ES_DATA_DIR="${ES_DATA_DIR:-$DEMO_ROOT/es-data}"
    export DEMO_N DEMO_SEED DEMO_ROOT
    echo "[--demo] DEMO_N=$DEMO_N DEMO_SEED=$DEMO_SEED DEMO_ROOT=$DEMO_ROOT"
fi
export DEMO

# r2c2_index — Indexation Pipeline

Self-contained corpus → ES + FAISS + KG indexation pipeline for the R2C2 agentic-RAG system. Reproduces the full set of indexes consumed by the runtime MCP server (`bitem-passages`, `bitem-passages-ner`, `bitem-documents`, `bitem-faiss-plain`, `bitem-faiss-enriched`, `bitem-kg.json`).

## Contents

One subpackage per stage, plus the SLURM templates that run them:

```
src/r2c2_index/
  corpus_parse/   Stage 1 — wikitext → ~200-word passages
  ner/            Stage 2 — LLM-driven NER + within/cross-doc refinement
  kg_build/       Stage 3 — entity co-occurrence KG (NPMI, prune NPMI≥0.1 ∧ count≥3)
  es_index/       Stage 4 — bulk-load 3 Elasticsearch indexes
  faiss_encode/   Stage 5 — sentence-transformer encode + FAISS IndexFlatIP (×2)
  slurm/          SLURM job templates (env-var-driven CHUNKS_DIR / INDEXES_DIR / ES_DATA_DIR)
```

Every stage is a Python package with a `__main__.py` that dispatches to the underlying sub-tool's `main()`:

```
python -m r2c2_index.corpus_parse <tool> [args ...]
python -m r2c2_index.ner          <tool> [args ...]
python -m r2c2_index.kg_build     <map|reduce> [args ...]
python -m r2c2_index.es_index     <tool> [args ...]
python -m r2c2_index.faiss_encode <tool> [args ...]
```

`python -m r2c2_index.<stage> --help` lists the available sub-tools. `python -m r2c2_index.<stage> <tool> --help` shows that tool's full original CLI (the dispatcher does not strip or rewrite any flags — every script's argparse is preserved verbatim).

## Required env vars (read by every sbatch)

Every SLURM template reads its locations from these variables, falling back to the default when one is unset:

| Var | Purpose | Default |
|---|---|---|
| `PROJECT` | Repo root | `$SLURM_SUBMIT_DIR` ‖ `$PWD` |
| `CORPUS_DIR` | Parent of `r2c2_corpus_v1.1/` (Stage 0 output) | `$PROJECT/data` |
| `CHUNKS_DIR` | Stage 1–2 output | `$PROJECT/data/chunks` |
| `INDEXES_DIR` | Stage 3–5 output (KG + FAISS) | `$PROJECT/data/indexes` |
| `ES_DATA_DIR` | Elasticsearch data dir for Stage 4 + worker.sbatch | `$PROJECT/data/es-data` |

## Stage order + canonical artefact paths

All paths below are resolved against `$CHUNKS_DIR` and `$INDEXES_DIR`. Set the env vars before invoking any sbatch.

| Order | Stage | Inputs | Outputs | Wall clock | SLURM template |
|-------|-------|--------|---------|------------|----------------|
| 1 | `corpus_parse.chunk_structured` | wikitext tar.gz archives in `$CORPUS_DIR/r2c2_corpus_v1.1/wikitext/` | `$CHUNKS_DIR/{passages.jsonl, documents.jsonl, corpus_meta.json}` | ~1–2 h | `slurm/run_pipeline.sbatch` |
| 2a | `ner.extract_entities_llm` | `$CHUNKS_DIR/passages.jsonl` + vLLM endpoint(s) | per-passage NER JSONL | 4–8 h | n/a (calls existing vLLM) |
| 2b | `ner.refine_entities` | NER JSONL + vLLM | within-doc-refined JSONL | 1–2 h | n/a |
| 2c | `ner.fast_crossdoc` / `parallel_crossdoc` | refined JSONL | cross-doc-refined JSONL | 30–60 min | n/a (multiprocess) |
| 2d | `ner.convert_ner_to_enriched` | cross-doc-refined JSONL | `$CHUNKS_DIR/passages_enriched.jsonl` | <10 min | n/a |
| 2e | `ner.merge_offsets_to_enriched` | enriched JSONL + offsets + `$CHUNKS_DIR/entity_type_map.json` | `$CHUNKS_DIR/passages_enriched_v3.jsonl` | ~15 min | `slurm/merge_offsets.sbatch` |
| 3 | `kg_build` (map+reduce) | `$CHUNKS_DIR/passages_enriched_v3.jsonl` | `$INDEXES_DIR/bitem-kg.json` | 20–30 min | `slurm/build_kg.sbatch` → `slurm/build_kg_reduce.sbatch` |
| 4a | `start_es.sbatch` | — | long-running ES service + `$PROJECT/src/r2c2_index/slurm/logs/.es_url` | (ongoing) | `slurm/start_es.sbatch` |
| 4b | `es_index.index_to_elasticsearch --mode base` | `$CHUNKS_DIR/passages.jsonl` | `bitem-passages` (~2 M docs) | 30–45 min | `slurm/index_es_base.sbatch` |
| 4c | `es_index.index_to_elasticsearch --mode ner` | `$CHUNKS_DIR/passages_enriched_v3.jsonl` | `bitem-passages-ner` (~2 M docs) | 45–60 min | `slurm/index_es_ner.sbatch` |
| 4d | `es_index.index_documents_to_es` | `$CHUNKS_DIR/documents.jsonl` | `bitem-documents` | ~5 min | `slurm/index_es_docs.sbatch` |
| 4e | `es_index.es_manage_index finalize` | — | force-merge segments | 5–10 min | `slurm/finalize_es.sbatch` |
| 5a | `faiss_encode.build_dense_index --index-type plain` | `$CHUNKS_DIR/passages.jsonl` | `$INDEXES_DIR/bitem-faiss-plain/` | 4–6 h on 1× A100 | `slurm/build_faiss_plain.sbatch` |
| 5b | `faiss_encode.build_dense_index --index-type enriched` | `$CHUNKS_DIR/passages_enriched_v3.jsonl` + `entity_type_map.json` | `$INDEXES_DIR/bitem-faiss-enriched/` | 4–6 h on 1× A100 | `slurm/build_faiss_enriched.sbatch` |

**Total wall clock** (all stages): ~10–12 h on a Baobab-class allocation (FAISS encoding dominates; NER + ES happen in parallel with FAISS via SLURM scheduling).

## CLI smoke

Every sub-tool exposes `--help` and parses argparse before doing any I/O, so each stage's CLI surface can be exercised against a fresh checkout:

```bash
uv run python -m r2c2_index.corpus_parse --help
uv run python -m r2c2_index.ner          --help
uv run python -m r2c2_index.kg_build     --help
uv run python -m r2c2_index.es_index     --help
uv run python -m r2c2_index.faiss_encode --help
```

Stages 2 (NER) and 4 (ES) need a running vLLM server and Elasticsearch instance respectively, so their full smoke tests live in the SLURM templates.

## Out of scope

- Calibration / tuning code (NER prompt tuning, judge agreement evaluation) is research-only and not included.
- Live full-corpus run is not part of this package; the indexes consumed at inference time are produced once via the SLURM templates here and referenced by the runtime MCP server.

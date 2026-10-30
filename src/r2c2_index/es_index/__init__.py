"""Stage 4: bulk-load three Elasticsearch indexes.

Sub-tools (each with its own ``main()``):
  - ``index_to_elasticsearch``   Bulk-load passages JSONL into
                                 ``bitem-passages`` (mode=base) or
                                 ``bitem-passages-ner`` (mode=ner).
  - ``index_documents_to_es``    Bulk-load documents JSONL into
                                 ``bitem-documents``.
  - ``es_manage_index``          Create / delete / force-merge index ops
                                 (BM25 k1=0.82, b=0.68, MS MARCO tuning).
"""

from __future__ import annotations

__all__ = [
    "index_to_elasticsearch",
    "index_documents_to_es",
    "es_manage_index",
]

"""Private helper modules for r2c2_index.ner.

These are not part of the CLI. The NER scripts add this directory to
``sys.path`` at import time and import the modules by bare name
(``from llm_utils import EndpointPool``, ``from entity_utils import ...``,
``from build_document_prompt import ...``).

Modules:
  - ``llm_utils``                vLLM endpoint pool / OpenAI client wrapper.
  - ``entity_utils``             Entity normalization + canonical type list.
  - ``build_document_prompt``    Document-level prompt builder + indexers.
  - ``document_reconstruction``  Reconstruct full doc text from passages.
  - ``span_utils``               Find entity spans + resolve overlaps.
"""

from __future__ import annotations

__all__: list[str] = []

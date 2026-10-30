"""CLI dispatcher for ``python -m r2c2_index.es_index <tool> [args...]``.

Forwards remaining argv to the chosen sub-tool's ``main()``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import es_manage_index, index_documents_to_es, index_to_elasticsearch

__all__ = ["main"]

_TOOLS: dict[str, Callable[[], None]] = {
    "index_to_elasticsearch": index_to_elasticsearch.main,
    "index_documents_to_es": index_documents_to_es.main,
    "es_manage_index": es_manage_index.main,
}


def _dispatcher_help() -> str:
    tools = "\n".join(f"  {name}" for name in sorted(_TOOLS))
    return (
        "usage: python -m r2c2_index.es_index <tool> [tool args ...]\n\n"
        "Stage 4: Elasticsearch bulk-load. Pick a tool, then pass its own\n"
        "--help for tool-specific options.\n\n"
        f"available tools:\n{tools}\n"
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatch to one of the ES sub-tools.

    Args:
        argv: Override for ``sys.argv[1:]``; defaults to ``sys.argv[1:]``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        sys.stdout.write(_dispatcher_help())
        sys.exit(0 if args else 2)
    tool = args[0]
    if tool not in _TOOLS:
        parser = argparse.ArgumentParser(prog="python -m r2c2_index.es_index")
        parser.add_argument("tool", choices=sorted(_TOOLS))
        parser.parse_args(args[:1])
        return  # unreachable
    rest = args[1:]
    sys.argv = [f"r2c2_index.es_index.{tool}", *rest]
    _TOOLS[tool]()


if __name__ == "__main__":
    main()

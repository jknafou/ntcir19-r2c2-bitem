"""CLI dispatcher for ``python -m r2c2_index.ner <tool> [args...]``.

The dispatcher forwards ``argv[2:]`` to the tool's own ``main()``, so each
tool keeps its own ``--help``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import (
    convert_ner_to_enriched,
    extract_entities_llm,
    fast_crossdoc,
    merge_offsets_to_enriched,
    parallel_crossdoc,
    refine_entities,
)

__all__ = ["main"]

_TOOLS: dict[str, Callable[[], None]] = {
    "extract_entities_llm": extract_entities_llm.main,
    "refine_entities": refine_entities.main,
    "fast_crossdoc": fast_crossdoc.main,
    "parallel_crossdoc": parallel_crossdoc.main,
    "convert_ner_to_enriched": convert_ner_to_enriched.main,
    "merge_offsets_to_enriched": merge_offsets_to_enriched.main,
}


def _dispatcher_help() -> str:
    tools = "\n".join(f"  {name}" for name in sorted(_TOOLS))
    return (
        "usage: python -m r2c2_index.ner <tool> [tool args ...]\n\n"
        "Stage 2: NER extraction + refinement. Pick a tool, then pass its\n"
        "own --help for tool-specific options.\n\n"
        f"available tools:\n{tools}\n"
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatch to one of the NER sub-tools."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        sys.stdout.write(_dispatcher_help())
        sys.exit(0 if args else 2)
    tool = args[0]
    if tool not in _TOOLS:
        parser = argparse.ArgumentParser(prog="python -m r2c2_index.ner")
        parser.add_argument("tool", choices=sorted(_TOOLS))
        parser.parse_args(args[:1])
        return  # unreachable
    rest = args[1:]
    sys.argv = [f"r2c2_index.ner.{tool}", *rest]
    _TOOLS[tool]()


if __name__ == "__main__":
    main()

"""CLI dispatcher for ``python -m r2c2_index.corpus_parse <tool> [args...]``.

The dispatcher forwards ``argv[2:]`` to the chosen tool's own ``main()``,
preserving each tool's full CLI (``--help`` works for both the
dispatcher and each sub-tool).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import chunk_structured, chunker

__all__ = ["main"]

_TOOLS: dict[str, Callable[[], None]] = {
    "chunk_structured": chunk_structured.main,
    "chunker": chunker.main,
}


def _dispatcher_help() -> str:
    """Return the dispatcher-level help text."""
    tools = "\n".join(f"  {name}" for name in sorted(_TOOLS))
    return (
        "usage: python -m r2c2_index.corpus_parse <tool> [tool args ...]\n\n"
        "Stage 1: corpus parse + chunk. Pick a tool, then pass its own --help\n"
        "for tool-specific options.\n\n"
        f"available tools:\n{tools}\n"
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatch to one of the corpus_parse sub-tools."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        sys.stdout.write(_dispatcher_help())
        sys.exit(0 if args else 2)
    tool = args[0]
    if tool not in _TOOLS:
        parser = argparse.ArgumentParser(prog="python -m r2c2_index.corpus_parse")
        parser.add_argument("tool", choices=sorted(_TOOLS))
        parser.parse_args(args[:1])
        return  # unreachable
    rest = args[1:]
    sys.argv = [f"r2c2_index.corpus_parse.{tool}", *rest]
    _TOOLS[tool]()


if __name__ == "__main__":
    main()

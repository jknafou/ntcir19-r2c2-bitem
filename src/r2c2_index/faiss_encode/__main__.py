"""CLI dispatcher for ``python -m r2c2_index.faiss_encode <tool> [args...]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import build_dense_index, merge_faiss_shards

__all__ = ["main"]

_TOOLS: dict[str, Callable[[], None]] = {
    "build_dense_index": build_dense_index.main,
    "merge_faiss_shards": merge_faiss_shards.main,
}


def _dispatcher_help() -> str:
    tools = "\n".join(f"  {name}" for name in sorted(_TOOLS))
    return (
        "usage: python -m r2c2_index.faiss_encode <tool> [tool args ...]\n\n"
        "Stage 5: FAISS dense-encoding. Pick a tool, then pass its own --help\n"
        "for tool-specific options.\n\n"
        f"available tools:\n{tools}\n"
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatch to one of the FAISS sub-tools.

    Args:
        argv: Override for ``sys.argv[1:]``; defaults to ``sys.argv[1:]``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        sys.stdout.write(_dispatcher_help())
        sys.exit(0 if args else 2)
    tool = args[0]
    if tool not in _TOOLS:
        parser = argparse.ArgumentParser(prog="python -m r2c2_index.faiss_encode")
        parser.add_argument("tool", choices=sorted(_TOOLS))
        parser.parse_args(args[:1])
        return  # unreachable
    rest = args[1:]
    sys.argv = [f"r2c2_index.faiss_encode.{tool}", *rest]
    _TOOLS[tool]()


if __name__ == "__main__":
    main()

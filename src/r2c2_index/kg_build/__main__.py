"""CLI dispatcher for ``python -m r2c2_index.kg_build``.

Single sub-tool (``build_kg``) is exposed directly: ``--help`` shows the
``build_kg.main`` argparse help so the dispatcher stays out of the way.
"""

from __future__ import annotations

import sys

from . import build_kg

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Run ``build_kg.main()`` on the provided CLI args.

    Args:
        argv: Override for ``sys.argv[1:]``; defaults to ``sys.argv[1:]``.
    """
    if argv is not None:
        sys.argv = ["r2c2_index.kg_build.build_kg", *argv]
    else:
        sys.argv = ["r2c2_index.kg_build.build_kg", *sys.argv[1:]]
    build_kg.main()


if __name__ == "__main__":
    main()

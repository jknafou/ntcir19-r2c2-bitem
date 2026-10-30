"""Module entry — delegates to ``r2c2_agent.cli.main``.

Run via::

    python -m r2c2_agent <subcommand> [options]
"""

from __future__ import annotations

from r2c2_agent.cli import main

__all__: list[str] = []


if __name__ == "__main__":
    raise SystemExit(main())

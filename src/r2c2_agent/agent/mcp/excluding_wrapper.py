"""Client-side MCP wrapper that enforces an exclusion set across every
tool that can return a passage id.

``ExcludingMCPWrapper`` decorates any :class:`MCPClientProtocol` and:

1. For ``search_passages``, injects an ``exclude_ids`` argument (a
   sorted, comma-separated list of passage ids) that the MCP server
   applies as a pre-retrieval filter across all three legs (BM25
   ``must_not`` on ES, KG ``must_not`` on ES, FAISS overfetch + mask).
2. For ``get_passage``, narrows the request to allowed pids and
   post-filters the response so excluded pids are replaced with an
   inline ``error`` row instead of the passage body. The agent
   therefore cannot read an excluded pid even if it happens to know
   its id (e.g. memorised it from a prior iteration's trace, or saw
   it leak through a corner-case retrieval bug).
3. For ``extract_from_passage``, an excluded ``passage_id``
   short-circuits to a "not available" response (no ``matches``).
4. For ``search_entities``, the response carries entity names + counts
   only — no pids — so it is a pure pass-through.

This is the foundation of the ``--ensemble N`` feature. Each ensemble
iteration K+1 is invoked with the union of pids the agent has touched
(retrieved, read, or judged) in iterations 0..K; the server treats
those pids as if they never existed in the index, and the wrapper
prevents any leak via lookup-by-pid tools. The same-``max_results``
contract to the LLM is preserved server-side — the wrapper never
post-filters ``search_passages`` (server-side ``must_not`` already
applied), only the lookup-by-pid tools.

Compatibility: a server that does not know ``exclude_ids`` accepts an
empty or absent value as a no-op, so the wrapper works against it with
an empty exclusion set. A server that honours the parameter applies the
exclusion with no change on the client side.

The orchestrator worker's ``process_question`` wraps its
``MCPHTTPClient`` instance in an ``ExcludingMCPWrapper`` between
ensemble iterations when the running ``touched_pids`` set is non-empty.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from r2c2_agent.agent.interfaces.mcp import MCPClientProtocol

__all__ = [
    "EXCLUDE_TOOL_NAME",
    "LOOKUP_TOOL_NAMES",
    "ExcludingMCPWrapper",
    "load_exclude_ids_from_file",
]

log = logging.getLogger(__name__)

# Tool that gets the exclude_ids injection (server-side filter).
EXCLUDE_TOOL_NAME = "search_passages"

# Lookup-by-pid tools whose responses must be post-filtered to drop any
# pid that is on the exclusion list. The MCP server has no exclusion
# semantics here (these tools are addressed by pid), so the wrapper is
# the only line of defence.
LOOKUP_TOOL_NAMES = frozenset({"get_passage", "extract_from_passage"})


def load_exclude_ids_from_file(path: Path) -> frozenset[str]:
    """Load an exclusion set from a newline-delimited passage-id file.

    Empty lines and lines beginning with ``#`` are ignored. Whitespace
    around each id is stripped. Returns a ``frozenset`` so the wrapper
    can cache it safely.

    Args:
        path: File containing one passage id per line.

    Returns:
        Frozen set of passage ids. Empty frozenset if the file is empty.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line)
    return frozenset(ids)


class ExcludingMCPWrapper:
    """An :class:`MCPClientProtocol` that enforces an exclude-pid set
    across every tool that can return a passage id.

    Forwards exclusion to the server's ``search_passages`` tool via a
    comma-separated ``exclude_ids`` argument; the server filters across
    all three retrieval legs. For ``get_passage`` and
    ``extract_from_passage`` (lookup-by-pid tools the server cannot
    pre-filter), the wrapper narrows the request and post-filters the
    response so excluded pids never reach the agent's context.

    The wrapper is constructed once per ensemble iteration with the
    union of pids the agent has touched in prior iterations. It is
    immutable — mutating the exclusion set mid-iteration would violate
    the per-iteration determinism the ensemble relies on.
    """

    def __init__(
        self,
        inner: MCPClientProtocol,
        exclude_ids: frozenset[str] | set[str],
    ) -> None:
        """Wrap ``inner`` and apply ``exclude_ids`` to every passage-bearing tool.

        Args:
            inner: The underlying MCP client (typically ``MCPHTTPClient``).
            exclude_ids: Passage ids to subtract from search and to
                blackhole on lookup. A frozenset is recommended; sets
                are accepted and frozen internally.
        """
        self._inner = inner
        self._exclude_ids: frozenset[str] = frozenset(exclude_ids)
        # Cached sorted CSV — rebuilt once at construction, reused on every
        # call_tool. Sorted so the wire representation is deterministic
        # for test assertions + trace replay.
        self._exclude_csv: str = ",".join(sorted(self._exclude_ids))

    @property
    def exclude_ids(self) -> frozenset[str]:
        """The passage ids enforced as excluded across every tool."""
        return self._exclude_ids

    @property
    def unwrapped(self) -> MCPClientProtocol:
        """The underlying client with no exclusion filter applied.

        Used by components that need corpus-wide retrieval (e.g. the
        finalizer's educated-guess verification step) — the wrapper's
        exclusion list is scoped to ensemble-diversity, not to truth
        verification.
        """
        return self._inner

    async def list_tools(self) -> list[dict[str, Any]]:
        """Delegate to the inner client without modification.

        The tool schemas exposed to the LLM are intentionally unchanged —
        the LLM never sees that exclusions are being applied. The
        server's ``exclude_ids`` parameter is a kwarg on the Python
        signature only; it is not registered in the LLM's tool schema.
        """
        return await self._inner.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Invoke a tool, enforcing the exclusion set across all surfaces.

        Three behaviours by tool:

        - ``search_passages``: inject ``exclude_ids=<sorted CSV>`` into
          ``arguments`` and forward. The server does all filtering on
          retrieval; this wrapper never post-filters its response.
        - ``get_passage`` / ``extract_from_passage``: forward (after
          narrowing the request to allowed pids), then post-filter the
          response so excluded pids are replaced with an inline error
          row. The agent therefore cannot read an excluded pid even if
          it knows the id.
        - All other tools (``search_entities``, etc.): pure pass-through.

        With an empty exclusion set the wrapper is a no-op forwarder.
        """
        if not self._exclude_ids:
            return await self._inner.call_tool(name, arguments, timeout=timeout)

        if name == EXCLUDE_TOOL_NAME:
            forwarded = dict(arguments)
            forwarded["exclude_ids"] = self._exclude_csv
            log.debug(
                "ExcludingMCPWrapper: forwarding %d excluded pids to search_passages",
                len(self._exclude_ids),
            )
            return await self._inner.call_tool(name, forwarded, timeout=timeout)

        if name == "extract_from_passage":
            # Short-circuit before the MCP call when the target pid is
            # excluded — saves latency and ES load. Shape mirrors the
            # server's "Passage not found" response so downstream
            # parsers stay uniform.
            target_pid = arguments.get("passage_id")
            if isinstance(target_pid, str) and target_pid in self._exclude_ids:
                log.debug(
                    "ExcludingMCPWrapper: short-circuit extract_from_passage(pid=%s)",
                    target_pid,
                )
                return {
                    "passage_id": target_pid,
                    "predicate": arguments.get("predicate", ""),
                    "error": "Passage excluded by ensemble feed-forward",
                    "matches": [],
                    "total_sentences_scanned": 0,
                    "too_broad": False,
                    "ensemble_excluded": True,
                }
            return await self._inner.call_tool(name, arguments, timeout=timeout)

        if name == "get_passage":
            # Filter the requested CSV so the server only loads pids the
            # agent is actually allowed to read. If every requested pid
            # is excluded, short-circuit with an all-error envelope.
            forwarded = dict(arguments)
            requested = arguments.get("passage_ids") or arguments.get("passage_id")
            requested_ids: list[str] = []
            if isinstance(requested, str):
                requested_ids = [
                    pid.strip() for pid in requested.split(",") if pid.strip()
                ]
            allowed_ids = [pid for pid in requested_ids if pid not in self._exclude_ids]
            blocked_ids = [pid for pid in requested_ids if pid in self._exclude_ids]

            if requested_ids and not allowed_ids:
                log.debug(
                    "ExcludingMCPWrapper: short-circuit get_passage; "
                    "all %d requested pids excluded",
                    len(requested_ids),
                )
                return {
                    "passages": [
                        {
                            "passage_id": pid,
                            "error": "Passage excluded by ensemble feed-forward",
                            "is_center": True,
                            "is_neighbor": False,
                            "ensemble_excluded": True,
                        }
                        for pid in requested_ids
                    ],
                    "count": len(requested_ids),
                    "window": int(arguments.get("window", 0) or 0),
                    "ensemble_excluded": True,
                }

            if blocked_ids and allowed_ids:
                # Mixed batch — narrow the request to allowed pids only.
                forwarded["passage_ids"] = ",".join(allowed_ids)
                if "passage_id" in forwarded:
                    forwarded["passage_id"] = ",".join(allowed_ids)
                log.debug(
                    "ExcludingMCPWrapper: get_passage narrowed %d→%d ids",
                    len(requested_ids),
                    len(allowed_ids),
                )

            raw = await self._inner.call_tool(name, forwarded, timeout=timeout)
            return self._post_filter_get_passage(raw, blocked_ids)

        # Any other tool (search_entities, etc.) is a pure pass-through —
        # they cannot return raw passage bodies.
        return await self._inner.call_tool(name, arguments, timeout=timeout)

    def _post_filter_get_passage(
        self,
        raw: Any,
        blocked_ids: list[str],
    ) -> dict[str, Any]:
        """Drop excluded pids from a ``get_passage`` response and append
        synthetic error rows for any blocked pid that was filtered out
        of the request before forwarding.

        MCP tools may return either a ``dict`` (already-parsed by the
        client) or a JSON string. Both shapes are accepted. Excluded
        pids are replaced with ``{passage_id, error, ensemble_excluded}``
        rows so the agent sees a unique, structured signal it can
        reflect on instead of silently missing chunks.
        """
        data: dict[str, Any]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw, "error": "invalid_json"}
            if not isinstance(parsed, dict):
                return {"raw": raw, "error": "non_object_response"}
            data = parsed
        elif isinstance(raw, dict):
            data = dict(raw)
        else:
            return {"error": "invalid_response_type", "raw": repr(raw)}

        passages = data.get("passages")
        if isinstance(passages, list):
            kept: list[Any] = []
            dropped = 0
            for entry in passages:
                if not isinstance(entry, dict):
                    kept.append(entry)
                    continue
                pid = entry.get("passage_id") or entry.get("pid")
                if isinstance(pid, str) and pid in self._exclude_ids:
                    dropped += 1
                    continue
                kept.append(entry)

            for blocked in blocked_ids:
                kept.append(
                    {
                        "passage_id": blocked,
                        "error": "Passage excluded by ensemble feed-forward",
                        "is_center": True,
                        "is_neighbor": False,
                        "ensemble_excluded": True,
                    }
                )
            data["passages"] = kept
            data["count"] = len(kept)
            if dropped or blocked_ids:
                data["ensemble_excluded_count"] = dropped + len(blocked_ids)
                log.debug(
                    "ExcludingMCPWrapper: get_passage filtered %d "
                    "server-rows + %d short-circuit-rows",
                    dropped,
                    len(blocked_ids),
                )
        elif isinstance(data.get("passage_id"), str):
            # Single-passage envelope (legacy shape).
            pid = data["passage_id"]
            if pid in self._exclude_ids:
                return {
                    "passage_id": pid,
                    "error": "Passage excluded by ensemble feed-forward",
                    "is_center": True,
                    "is_neighbor": False,
                    "ensemble_excluded": True,
                }

        return data

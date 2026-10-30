"""Tests for ``ExcludingMCPWrapper`` (ensemble feed-forward infrastructure).

The wrapper is a pure client-side decorator that forwards an ``exclude_ids``
kwarg to the MCP server's ``search_passages`` tool. All filtering happens
server-side across the 3 retrieval legs (BM25/KG ``must_not`` on ES, FAISS
overfetch + mask) — the wrapper does not post-filter responses.

Every test uses a :class:`FakeMCPClient` that records calls and returns
canned responses so we can pin down the forwarder semantics without
running a real MCP server.
"""

from __future__ import annotations

from typing import Any

import pytest

from r2c2_agent.agent.interfaces.mcp import MCPClientProtocol
from r2c2_agent.agent.mcp.excluding_wrapper import (
    EXCLUDE_TOOL_NAME,
    ExcludingMCPWrapper,
    load_exclude_ids_from_file,
)

class FakeMCPClient:
    """Minimal MCP client double that records calls and returns canned data.

    Satisfies :class:`MCPClientProtocol` structurally (duck typing) so
    ``ExcludingMCPWrapper`` treats it exactly like a real client.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": EXCLUDE_TOOL_NAME, "description": "stub"}]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        self.calls.append((name, dict(arguments), timeout))
        return dict(self.response)


def _make_passages(ids: list[str]) -> list[dict[str, Any]]:
    """Build a list of passage dicts that mimic the real MCP response shape."""
    return [
        {
            "passage_id": pid,
            "doc_id": pid.split("__")[0],
            "title": f"title-{pid}",
            "text": f"text-{pid}",
            "score": 1.0 - 0.01 * i,
            "rank": i + 1,
        }
        for i, pid in enumerate(ids)
    ]


def _make_response(ids: list[str]) -> dict[str, Any]:
    return {
        "main_query": "test",
        "queries": ["test"],
        "corpus": "all",
        "method": "hybrid",
        "total_results": len(ids),
        "passages": _make_passages(ids),
    }


# Constructor + basic properties


class TestConstruction:
    def test_empty_exclude_set(self) -> None:
        wrapper = ExcludingMCPWrapper(FakeMCPClient({}), set())
        assert wrapper.exclude_ids == frozenset()

    def test_set_input_is_frozen(self) -> None:
        exclude: set[str] = {"P1", "P2"}
        wrapper = ExcludingMCPWrapper(FakeMCPClient({}), exclude)
        assert wrapper.exclude_ids == frozenset({"P1", "P2"})
        # Mutating the original set must not affect the wrapper.
        exclude.add("P3")
        assert "P3" not in wrapper.exclude_ids

    def test_wrapper_satisfies_protocol(self) -> None:
        """Structural subtype check — wrapper is an MCPClientProtocol."""
        wrapper = ExcludingMCPWrapper(FakeMCPClient({}), frozenset())
        assert isinstance(wrapper, MCPClientProtocol)


# Pass-through paths (empty set, non-search tools)


class TestPassThrough:
    @pytest.mark.asyncio
    async def test_empty_exclude_set_passes_through(self) -> None:
        inner = FakeMCPClient(_make_response(["P1", "P2", "P3"]))
        wrapper = ExcludingMCPWrapper(inner, frozenset())

        await wrapper.call_tool(EXCLUDE_TOOL_NAME, {"queries": "q", "max_results": 20})

        # Empty set → no exclude_ids injected.
        assert "exclude_ids" not in inner.calls[0][1]

    @pytest.mark.asyncio
    async def test_non_search_tool_passes_through(self) -> None:
        inner = FakeMCPClient({"doc_id": "Q1", "text": "foo"})
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P1"}))

        result = await wrapper.call_tool("get_passage", {"passage_id": "Q1"})

        assert result == {"doc_id": "Q1", "text": "foo"}
        # The wrapper must not rewrite arguments for non-search tools.
        assert inner.calls[0][1] == {"passage_id": "Q1"}

    @pytest.mark.asyncio
    async def test_list_tools_passes_through(self) -> None:
        inner = FakeMCPClient({})
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P1"}))
        tools = await wrapper.list_tools()
        # The wrapper must not leak the exclude_ids parameter into the
        # tool schema the LLM sees.
        assert tools == [{"name": EXCLUDE_TOOL_NAME, "description": "stub"}]


# Forwarder semantics — exclude_ids is injected into arguments as sorted CSV


class TestForwarding:
    @pytest.mark.asyncio
    async def test_injects_exclude_ids_as_sorted_csv(self) -> None:
        inner = FakeMCPClient(_make_response([]))
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P3", "P1", "P2"}))

        await wrapper.call_tool(EXCLUDE_TOOL_NAME, {"queries": "q", "max_results": 20})

        # Sorted so the wire representation is deterministic across runs.
        assert inner.calls[0][1]["exclude_ids"] == "P1,P2,P3"

    @pytest.mark.asyncio
    async def test_server_response_is_not_post_filtered(self) -> None:
        """The wrapper must NOT touch the response. Filtering is server-side.
        Test: server returns an 'already-excluded' pid (simulating a bug or
        stale server); wrapper must pass it through verbatim, not drop it."""
        inner = FakeMCPClient(_make_response(["P1", "P_excluded", "P3"]))
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P_excluded"}))

        result = await wrapper.call_tool(
            EXCLUDE_TOOL_NAME, {"queries": "q", "max_results": 20}
        )

        # The wrapper forwards and does not filter, so the excluded pid
        # comes back even though a real server would not return it.
        assert [p["passage_id"] for p in result["passages"]] == [
            "P1",
            "P_excluded",
            "P3",
        ]
        assert result["total_results"] == 3

    @pytest.mark.asyncio
    async def test_preserves_other_arguments(self) -> None:
        inner = FakeMCPClient(_make_response([]))
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P1"}))

        await wrapper.call_tool(
            EXCLUDE_TOOL_NAME,
            {"queries": "foo", "corpus": "wookieepedia", "max_results": 50},
        )

        args = inner.calls[0][1]
        assert args["queries"] == "foo"
        assert args["corpus"] == "wookieepedia"
        assert args["max_results"] == 50
        assert args["exclude_ids"] == "P1"

    @pytest.mark.asyncio
    async def test_original_args_not_mutated(self) -> None:
        """The wrapper must not mutate the caller's arguments dict."""
        inner = FakeMCPClient(_make_response([]))
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P1"}))
        args = {"queries": "q", "max_results": 20}
        args_before = dict(args)

        await wrapper.call_tool(EXCLUDE_TOOL_NAME, args)

        assert args == args_before
        assert "exclude_ids" not in args

    @pytest.mark.asyncio
    async def test_timeout_forwarded(self) -> None:
        inner = FakeMCPClient(_make_response([]))
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P1"}))

        await wrapper.call_tool(EXCLUDE_TOOL_NAME, {"queries": "q"}, timeout=45.0)

        assert inner.calls[0][2] == 45.0


class TestLoadExcludeIds:
    def test_basic_load(self, tmp_path: Any) -> None:
        path = tmp_path / "excl.txt"
        path.write_text("Q1__00001\nQ2__00003\nQ5__00002\n")
        assert load_exclude_ids_from_file(path) == frozenset(
            {"Q1__00001", "Q2__00003", "Q5__00002"}
        )

    def test_skips_blanks_and_comments(self, tmp_path: Any) -> None:
        path = tmp_path / "excl.txt"
        path.write_text(
            "# This is a comment\nQ1__00001\n\n  # indented comment\n   Q2__00003   \n"
        )
        assert load_exclude_ids_from_file(path) == frozenset({"Q1__00001", "Q2__00003"})

    def test_empty_file_yields_empty_frozenset(self, tmp_path: Any) -> None:
        path = tmp_path / "excl.txt"
        path.write_text("")
        assert load_exclude_ids_from_file(path) == frozenset()

    def test_missing_file_raises(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            load_exclude_ids_from_file(tmp_path / "does_not_exist.txt")


class TestMultiIterationSimulation:
    """Simulate the ensemble workflow in memory: each iteration's touched
    pids become the next iteration's exclusion set. Verify the wrapper
    correctly forwards the growing CSV to the server."""

    @pytest.mark.asyncio
    async def test_three_iteration_exclusion_grows(self) -> None:
        # Iter 0: empty exclusion. Server returns P1..P5.
        inner0 = FakeMCPClient(_make_response(["P1", "P2", "P3", "P4", "P5"]))
        iter0 = ExcludingMCPWrapper(inner0, frozenset())
        await iter0.call_tool(EXCLUDE_TOOL_NAME, {"queries": "q"})
        assert "exclude_ids" not in inner0.calls[0][1]

        # Iter 1: exclude iter 0's touched pids.
        inner1 = FakeMCPClient(_make_response(["P6", "P7", "P8"]))
        iter1 = ExcludingMCPWrapper(inner1, frozenset({"P1", "P2", "P3"}))
        await iter1.call_tool(EXCLUDE_TOOL_NAME, {"queries": "q"})
        assert inner1.calls[0][1]["exclude_ids"] == "P1,P2,P3"

        # Iter 2: exclude iter 0+1's touched pids (union).
        inner2 = FakeMCPClient(_make_response(["P9", "P10", "P11"]))
        iter2 = ExcludingMCPWrapper(
            inner2, frozenset({"P1", "P2", "P3", "P6", "P7", "P8"})
        )
        await iter2.call_tool(EXCLUDE_TOOL_NAME, {"queries": "q"})
        assert inner2.calls[0][1]["exclude_ids"] == "P1,P2,P3,P6,P7,P8"


# Lookup-by-pid filtering


class TestGetPassageFiltering:
    """``get_passage`` short-circuits / narrows / post-filters when pids
    are on the exclusion list. The MCP server is addressed by pid here,
    so it cannot pre-filter — the wrapper is the only line of defence.
    """

    @pytest.mark.asyncio
    async def test_excluded_single_pid_short_circuits(self) -> None:
        """One requested pid, on exclusion list → no server round-trip."""
        inner = FakeMCPClient({"passages": [{"passage_id": "X1", "text": "BAD"}]})
        wrapper = ExcludingMCPWrapper(inner, frozenset({"X1"}))

        result = await wrapper.call_tool("get_passage", {"passage_ids": "X1"})

        # The server is not called: the wrapper short-circuits before the round-trip.
        assert inner.calls == []
        assert result["count"] == 1
        assert result["passages"][0]["passage_id"] == "X1"
        assert result["passages"][0]["error"]
        assert result["passages"][0]["ensemble_excluded"] is True
        assert "text" not in result["passages"][0]

    @pytest.mark.asyncio
    async def test_mixed_batch_narrows_request_and_appends_errors(self) -> None:
        """Some excluded, some allowed → server gets only the allowed
        pids, response carries error rows for the blocked ones."""
        inner = FakeMCPClient(
            {
                "passages": [
                    {"passage_id": "P_OK", "text": "good"},
                ],
                "count": 1,
            }
        )
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P_BAD"}))

        result = await wrapper.call_tool("get_passage", {"passage_ids": "P_BAD,P_OK"})

        # Server saw only the allowed pid.
        assert inner.calls[0][1]["passage_ids"] == "P_OK"
        # Response carries both: real row for P_OK, synthetic error for P_BAD.
        pids = {p["passage_id"]: p for p in result["passages"]}
        assert pids["P_OK"]["text"] == "good"
        assert pids["P_BAD"]["error"]
        assert pids["P_BAD"]["ensemble_excluded"] is True
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_post_filter_drops_excluded_pid_returned_by_server(self) -> None:
        """Even if the server (buggy / stale) returns an excluded pid in
        ``passages``, the wrapper drops it from the response."""
        inner = FakeMCPClient(
            {
                "passages": [
                    {"passage_id": "OK", "text": "good"},
                    {"passage_id": "LEAK", "text": "should-be-dropped"},
                ],
                "count": 2,
            }
        )
        wrapper = ExcludingMCPWrapper(inner, frozenset({"LEAK"}))

        # Request pid "OK" only — server-side leak of "LEAK" must still be filtered.
        result = await wrapper.call_tool("get_passage", {"passage_ids": "OK"})

        pids = [p["passage_id"] for p in result["passages"]]
        assert "LEAK" not in pids
        assert "OK" in pids
        assert result.get("ensemble_excluded_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_json_string_response_is_parsed_and_filtered(self) -> None:
        """MCP servers may return JSON-encoded strings (legacy shape).
        The wrapper must parse, filter, and return a dict.
        """
        import json as _json

        class _StringResponseClient:
            """Returns the JSON string verbatim — mirrors the real MCP
            transport, which JSON-encodes everything."""

            def __init__(self, payload: str) -> None:
                self._payload = payload
                self.calls: list[tuple[str, dict[str, Any], float]] = []

            async def list_tools(self) -> list[dict[str, Any]]:
                return []

            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any],
                timeout: float = 30.0,
            ) -> Any:
                self.calls.append((name, dict(arguments), timeout))
                return self._payload  # type: ignore[return-value]

        raw_str = _json.dumps(
            {
                "passages": [
                    {"passage_id": "OK", "text": "good"},
                    {"passage_id": "LEAK", "text": "bad"},
                ],
                "count": 2,
            }
        )
        inner = _StringResponseClient(raw_str)
        wrapper = ExcludingMCPWrapper(inner, frozenset({"LEAK"}))

        result = await wrapper.call_tool("get_passage", {"passage_ids": "OK"})

        pids = [p["passage_id"] for p in result["passages"]]
        assert "LEAK" not in pids
        assert "OK" in pids


class TestExtractFromPassageFiltering:
    """``extract_from_passage`` short-circuits when the target pid is
    on the exclusion list — no MCP round-trip, no NLI compute."""

    @pytest.mark.asyncio
    async def test_excluded_pid_short_circuits(self) -> None:
        inner = FakeMCPClient({"passage_id": "X", "matches": [{"text": "secret"}]})
        wrapper = ExcludingMCPWrapper(inner, frozenset({"X"}))

        result = await wrapper.call_tool(
            "extract_from_passage",
            {"passage_id": "X", "predicate": "what is secret"},
        )

        # The server is not called.
        assert inner.calls == []
        assert result["passage_id"] == "X"
        assert result["matches"] == []
        assert result["ensemble_excluded"] is True

    @pytest.mark.asyncio
    async def test_allowed_pid_passes_through(self) -> None:
        inner = FakeMCPClient(
            {"passage_id": "OK", "matches": [{"text": "fact"}], "too_broad": False}
        )
        wrapper = ExcludingMCPWrapper(inner, frozenset({"X"}))

        result = await wrapper.call_tool(
            "extract_from_passage",
            {"passage_id": "OK", "predicate": "what fact"},
        )

        assert inner.calls[0][0] == "extract_from_passage"
        assert result["matches"] == [{"text": "fact"}]


class TestPassThroughEntities:
    """``search_entities`` returns entity names + counts, no pids — pure
    pass-through with the exclusion set non-empty."""

    @pytest.mark.asyncio
    async def test_search_entities_unaffected(self) -> None:
        inner = FakeMCPClient({"entities": [{"entity": "Vader", "passage_count": 5}]})
        wrapper = ExcludingMCPWrapper(inner, frozenset({"P1", "P2"}))

        result = await wrapper.call_tool("search_entities", {"query": "Vader"})

        # No mutation, no exclude_ids on the args.
        assert "exclude_ids" not in inner.calls[0][1]
        assert result == {"entities": [{"entity": "Vader", "passage_count": 5}]}

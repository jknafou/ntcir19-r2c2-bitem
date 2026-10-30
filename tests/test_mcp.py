"""Tests for MCP clients (MCPHTTPClient mock + DummyMCPClient)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from r2c2_agent.agent.config import NER_TOOLS
from r2c2_agent.agent.exceptions import MCPError
from r2c2_agent.agent.mcp.client import MCPHTTPClient, mcp_tool_to_openai
from r2c2_agent.agent.mcp.dummy import DUMMY_PASSAGES, DummyMCPClient


class TestMCPToolConversion:
    def test_basic_conversion(self) -> None:
        mcp_tool = {
            "name": "search_passages",
            "description": "Search for passages",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        }
        result = mcp_tool_to_openai(mcp_tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search_passages"
        assert result["function"]["description"] == "Search for passages"
        assert "query" in result["function"]["parameters"]["properties"]

    def test_missing_description(self) -> None:
        mcp_tool = {"name": "test_tool", "inputSchema": {"type": "object"}}
        result = mcp_tool_to_openai(mcp_tool)
        assert result["function"]["description"] == ""

    def test_missing_schema(self) -> None:
        mcp_tool = {"name": "test_tool"}
        result = mcp_tool_to_openai(mcp_tool)
        assert result["function"]["parameters"]["type"] == "object"


# MCPHTTPClient against mocked SDK sessions


def _make_mock_tool(
    name: str = "search_passages",
    description: str = "Search",
    input_schema: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock MCP Tool object matching the SDK's Tool type."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
    return tool


def _make_text_content(text: str) -> MagicMock:
    """Create a mock TextContent matching the SDK's TextContent type."""
    from mcp.types import TextContent

    content = MagicMock(spec=TextContent)
    content.text = text
    # Make isinstance(content, TextContent) work
    content.__class__ = TextContent
    return content


def _make_call_result(texts: list[str], is_error: bool = False) -> MagicMock:
    """Create a mock CallToolResult."""
    result = MagicMock()
    result.content = [_make_text_content(t) for t in texts]
    result.isError = is_error
    return result


def _make_list_result(tools: list[MagicMock]) -> MagicMock:
    """Create a mock ListToolsResult."""
    result = MagicMock()
    result.tools = tools
    return result


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock ClientSession with list_tools and call_tool."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=_make_list_result([_make_mock_tool()]))
    session.call_tool = AsyncMock(
        return_value=_make_call_result([json.dumps({"passages": [{"id": "p1"}]})])
    )
    return session


def _mock_streamable_http(session: AsyncMock) -> Any:
    """Create a mock streamablehttp_client context manager that yields the given session.

    The mock replaces both ``streamablehttp_client`` and ``ClientSession``
    so that any ``async with streamablehttp_client(url)`` / ``async with
    ClientSession(...)`` pair yields the provided mock session.
    """
    read_stream = MagicMock()
    write_stream = MagicMock()

    @asynccontextmanager
    async def _mock_client(url: str, **kwargs: Any) -> Any:
        yield read_stream, write_stream, None

    return _mock_client, session


async def _create_connected_client(
    mock_session: AsyncMock,
    standard_url: str = "http://test:9000/mcp",
    ner_url: str = "",
    ner_session: AsyncMock | None = None,
) -> MCPHTTPClient:
    """Create an MCPHTTPClient marked as connected, with mocked sessions.

    The client opens a session per call and keeps none between calls, so
    only the ``_connected`` flag is set.  Callers must patch
    ``streamablehttp_client`` and ``ClientSession`` to inject mock sessions.
    """
    client = MCPHTTPClient(standard_url=standard_url, ner_url=ner_url)
    client._connected = True
    if ner_url and ner_session is not None:
        client._ner_available = True
    return client


class TestMCPHTTPClient:
    @pytest.mark.asyncio
    async def test_list_tools(self, mock_session: AsyncMock) -> None:
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            result = await client.list_tools()

        assert len(result) == 1
        assert result[0]["function"]["name"] == "search_passages"
        assert result[0]["type"] == "function"

    @pytest.mark.asyncio
    async def test_list_tools_cached(self, mock_session: AsyncMock) -> None:
        """Second call uses cache, no SDK request."""
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            r1 = await client.list_tools()
            r2 = await client.list_tools()

        assert r1 == r2
        # Only one SDK call (second was cached)
        mock_session.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_tool_text_content(self, mock_session: AsyncMock) -> None:
        """Tools return text content; client extracts and parses JSON."""
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            result = await client.call_tool("search_passages", {"query": "test"})

        assert "passages" in result

    @pytest.mark.asyncio
    async def test_call_tool_non_json_text(self, mock_session: AsyncMock) -> None:
        """Non-JSON text content is returned as {'text': ...}."""
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result(["plain text result"])
        )
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            result = await client.call_tool("search_passages", {"query": "test"})

        assert result == {"text": "plain text result"}

    @pytest.mark.asyncio
    async def test_call_tool_sdk_error(self, mock_session: AsyncMock) -> None:
        """SDK McpError returns error dict instead of raising."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        mock_session.call_tool = AsyncMock(
            side_effect=McpError(ErrorData(code=-1, message="timeout"))
        )
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            result = await client.call_tool("search_passages", {"query": "test"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_call_tool_generic_error(self, mock_session: AsyncMock) -> None:
        """Unexpected exceptions return error dict."""
        mock_session.call_tool = AsyncMock(
            side_effect=ConnectionError("connection reset")
        )
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            result = await client.call_tool("search_passages", {"query": "test"})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        """Operations before connect() raise MCPError."""
        client = MCPHTTPClient(standard_url="http://test:9000/mcp")

        with pytest.raises(MCPError, match="not connected"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_ner_routing(self) -> None:
        """NER tools are routed to the NER session."""
        standard_session = AsyncMock()
        standard_session.initialize = AsyncMock()
        standard_session.call_tool = AsyncMock(
            return_value=_make_call_result([json.dumps({"result": "standard"})])
        )
        standard_session.__aenter__ = AsyncMock(return_value=standard_session)
        standard_session.__aexit__ = AsyncMock(return_value=False)

        ner_session = AsyncMock()
        ner_session.initialize = AsyncMock()
        ner_session.call_tool = AsyncMock(
            return_value=_make_call_result([json.dumps({"result": "ner"})])
        )
        ner_session.__aenter__ = AsyncMock(return_value=ner_session)
        ner_session.__aexit__ = AsyncMock(return_value=False)

        client = MCPHTTPClient(
            standard_url="http://standard:9000/mcp",
            ner_url="http://ner:9001/mcp",
        )
        client._connected = True
        client._ner_available = True

        session_map = {
            "http://standard:9000/mcp": standard_session,
            "http://ner:9001/mcp": ner_session,
        }

        @asynccontextmanager
        async def _mock_client(url: str, **kwargs: Any) -> Any:
            yield MagicMock(), MagicMock(), None

        def _mock_session_factory(read: Any, write: Any) -> Any:
            # Determine which session based on call order
            return _mock_session_factory._next_session

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", _mock_client),
        ):
            # Standard tool call
            _mock_session_factory._next_session = standard_session
            with patch(
                "r2c2_agent.agent.mcp.client.ClientSession",
                return_value=standard_session,
            ):
                result1 = await client.call_tool("search_passages", {"query": "test"})
            standard_session.call_tool.assert_awaited_once()
            assert result1["result"] == "standard"

            # NER tool call
            with patch(
                "r2c2_agent.agent.mcp.client.ClientSession",
                return_value=ner_session,
            ):
                result2 = await client.call_tool("search_by_entity", {"entity": "test"})
            ner_session.call_tool.assert_awaited_once()
            assert result2["result"] == "ner"

    @pytest.mark.asyncio
    async def test_ner_tools_explicit_set(self) -> None:
        """NER routing uses the explicit NER_TOOLS set."""
        assert "search_by_entity" in NER_TOOLS
        assert "search_entities" in NER_TOOLS
        assert "search_passages" not in NER_TOOLS

    @pytest.mark.asyncio
    async def test_close_resets_state(self, mock_session: AsyncMock) -> None:
        """close() resets connected state and clears cache."""
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            await client.list_tools()
        assert client._tools_cache is not None

        await client.close()
        assert not client._connected
        assert client._tools_cache is None

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, mock_session: AsyncMock) -> None:
        """Calling connect() on an already-connected client is a no-op."""
        client = await _create_connected_client(mock_session)
        # Should not raise
        await client.connect()
        assert client._connected

    @pytest.mark.asyncio
    async def test_call_tool_error_result(self, mock_session: AsyncMock) -> None:
        """isError=True with no text content returns error dict."""
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result([], is_error=True)
        )
        client = await _create_connected_client(mock_session)
        mock_client_cm, _ = _mock_streamable_http(mock_session)

        with (
            patch("r2c2_agent.agent.mcp.client.streamablehttp_client", mock_client_cm),
            patch("r2c2_agent.agent.mcp.client.ClientSession", return_value=mock_session),
        ):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            result = await client.call_tool("search_passages", {"query": "test"})

        assert "error" in result


class TestDummyMCPClient:
    @pytest.mark.asyncio
    async def test_list_tools_returns_schemas(self) -> None:
        client = DummyMCPClient()
        tools = await client.list_tools()
        assert len(tools) == 3
        names = {t["function"]["name"] for t in tools}
        assert names == {"search_passages", "get_passage", "get_document"}

    @pytest.mark.asyncio
    async def test_list_tools_openai_format(self) -> None:
        client = DummyMCPClient()
        tools = await client.list_tools()
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_search_passages(self) -> None:
        client = DummyMCPClient()
        result = await client.call_tool("search_passages", {"query": "Jaws director"})
        assert "passages" in result
        assert len(result["passages"]) > 0
        # Jaws passage should be found
        pids = [p["passage_id"] for p in result["passages"]]
        assert "dummy_p1" in pids

    @pytest.mark.asyncio
    async def test_search_fallback(self) -> None:
        """Query matching nothing returns first 3 passages as fallback."""
        client = DummyMCPClient()
        result = await client.call_tool("search_passages", {"query": "zzzzzzz"})
        assert len(result["passages"]) == 3

    @pytest.mark.asyncio
    async def test_get_passage(self) -> None:
        client = DummyMCPClient()
        result = await client.call_tool("get_passage", {"passage_id": "dummy_p1"})
        assert result["passage_id"] == "dummy_p1"
        assert "Spielberg" in result["text"]

    @pytest.mark.asyncio
    async def test_get_passage_not_found(self) -> None:
        client = DummyMCPClient()
        result = await client.call_tool("get_passage", {"passage_id": "nonexistent"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_document(self) -> None:
        client = DummyMCPClient()
        result = await client.call_tool("get_document", {"doc_id": "Q567890"})
        assert result["doc_id"] == "Q567890"
        assert "Streep" in result["title"]
        assert len(result["passages"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_tool(self) -> None:
        client = DummyMCPClient()
        result = await client.call_tool("nonexistent_tool", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_passage_fields(self) -> None:
        """All passages have required fields."""
        for p in DUMMY_PASSAGES:
            assert "passage_id" in p
            assert "doc_id" in p
            assert "title" in p
            assert "section" in p
            assert "text" in p

    @pytest.mark.asyncio
    async def test_top_k_limit(self) -> None:
        client = DummyMCPClient()
        result = await client.call_tool(
            "search_passages", {"query": "film", "top_k": 2}
        )
        assert len(result["passages"]) <= 2

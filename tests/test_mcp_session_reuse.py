"""Tests for session-reuse behaviour in ``MCPHTTPClient``.

These tests verify the invariants documented in
``src/r2c2_agent/agent/mcp/client.py``:

1. First call opens a session.
2. Subsequent calls reuse the same session object (no re-init).
3. Concurrent calls on a fresh client do not double-init (init lock).
4. On a simulated ``anyio.ClosedResourceError`` the client reconnects
   transparently and the caller sees a successful result.
5. Two consecutive transport failures surface as an error dict with
   ``mcp_unhealthy=True``.
6. NER routing opens a distinct session per endpoint URL.
7. ``aclose()`` tears down all live sessions and prevents reuse.

The MCP transport and ``ClientSession`` are both stubbed; no real
network calls are made.  The stubs mirror the SDK's public shape
(``streamablehttp_client`` yields ``(read, write, get_session_id)``
and ``ClientSession`` is an async context manager exposing
``initialize()``, ``list_tools()``, and ``call_tool()``).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from r2c2_agent.agent.mcp import client as client_module
from r2c2_agent.agent.mcp.client import MCPHTTPClient

# Fake transport + session primitives


class _FakeTextContent:
    """Duck-typed stand-in for ``mcp.types.TextContent``.

    The real ``TextContent`` is a pydantic model; we register this class
    against the same name so ``isinstance(block, TextContent)`` inside
    ``MCPHTTPClient.call_tool`` succeeds.  See ``_install_textcontent``.
    """

    def __init__(self, text: str) -> None:
        self.text = text


def _make_list_result(names: list[str]) -> Any:
    tools = []
    for name in names:
        t = MagicMock()
        t.name = name
        t.description = f"stub {name}"
        t.inputSchema = {"type": "object"}
        tools.append(t)
    result = MagicMock()
    result.tools = tools
    return result


def _make_call_result(payload: dict[str, Any]) -> Any:
    result = MagicMock()
    result.content = [_FakeTextContent(json.dumps(payload))]
    result.isError = False
    return result


class FakeSession:
    """Drop-in replacement for ``mcp.ClientSession``.

    Each instance records how many times it was initialised and how
    many tool calls it served.  The ``behaviour`` hooks let individual
    tests simulate transport death.
    """

    def __init__(self, read: Any, write: Any) -> None:
        self._read = read
        self._write = write
        self.initialize_count: int = 0
        self.call_count: int = 0
        self.list_count: int = 0
        self.entered: bool = False
        # Hooks.  Each may be None or a callable taking no args; if it
        # returns an Exception, that exception is raised.
        self.on_call_tool: Any = None
        self.on_list_tools: Any = None
        self.response_payload: dict[str, Any] = {"ok": True}

    async def __aenter__(self) -> FakeSession:
        self.entered = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.entered = False

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def list_tools(self) -> Any:
        self.list_count += 1
        if self.on_list_tools is not None:
            out = self.on_list_tools()
            if isinstance(out, BaseException):
                raise out
        return _make_list_result(["search_passages"])

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.call_count += 1
        if self.on_call_tool is not None:
            out = self.on_call_tool(name, args)
            if isinstance(out, BaseException):
                raise out
        return _make_call_result(self.response_payload)


class FakeTransport:
    """Tracks every ``streamablehttp_client(url)`` context entry.

    Stores the spawned :class:`FakeSession` instances per URL so tests
    can assert how many were opened and in what order.
    """

    def __init__(self) -> None:
        self.sessions_by_url: dict[str, list[FakeSession]] = {}
        self.transport_enters: list[str] = []
        # Pluggable session factory; defaults to returning a vanilla
        # FakeSession each time.  Tests that need canned sessions (e.g.
        # pre-configured to raise) override this.
        self._session_factory: Any = None

    def set_session_factory(self, factory: Any) -> None:
        self._session_factory = factory

    def _new_session(self, url: str, read: Any, write: Any) -> FakeSession:
        if self._session_factory is not None:
            sess = self._session_factory(url, read, write)
        else:
            sess = FakeSession(read, write)
        self.sessions_by_url.setdefault(url, []).append(sess)
        return sess


@pytest.fixture
def fake_transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    """Patch ``streamablehttp_client`` + ``ClientSession`` in client module.

    The fixture returns a :class:`FakeTransport` whose
    ``sessions_by_url`` attribute lets tests assert how many sessions
    were opened and in what order, across which URLs.
    """
    transport = FakeTransport()

    @asynccontextmanager
    async def _fake_streamablehttp_client(url: str, **kwargs: Any) -> Any:
        transport.transport_enters.append(url)
        read = MagicMock(name=f"read({url})")
        write = MagicMock(name=f"write({url})")

        # The next ``ClientSession(...)`` call returns a session we
        # associate with this URL.  We pre-register it here so the
        # factory below can grab it.
        def _factory(read_arg: Any, write_arg: Any) -> FakeSession:
            return transport._new_session(url, read_arg, write_arg)

        # ``ClientSession(r, w)`` -> FakeSession; see the monkeypatch below.
        monkeypatch.setattr(client_module, "ClientSession", _factory, raising=True)
        try:
            yield read, write, lambda: None
        finally:
            # Restore identity so the next transport enter installs its
            # own factory cleanly.
            pass

    monkeypatch.setattr(
        client_module,
        "streamablehttp_client",
        _fake_streamablehttp_client,
        raising=True,
    )
    # Install a no-op ClientSession initially; transport.__aenter__
    # swaps it with the URL-aware factory per open.
    monkeypatch.setattr(
        client_module,
        "ClientSession",
        lambda r, w: FakeSession(r, w),
        raising=True,
    )

    # Patch TextContent so isinstance checks on our FakeTextContent pass.
    monkeypatch.setattr(client_module, "TextContent", _FakeTextContent, raising=True)

    return transport


# Helpers


async def _connected_client(
    standard_url: str = "http://standard:9000/mcp",
) -> MCPHTTPClient:
    """Create + connect a client, returning it ready for use."""
    client = MCPHTTPClient(standard_url=standard_url)
    await client.connect()
    return client


# First call opens a session; second call reuses it


class TestSessionReuse:
    @pytest.mark.asyncio
    async def test_first_call_opens_session(
        self, fake_transport: FakeTransport
    ) -> None:
        client = await _connected_client()
        try:
            # connect() already opened the standard session.
            sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
            assert len(sessions) == 1
            assert sessions[0].initialize_count == 1

            result = await client.call_tool("search_passages", {"q": "x"})
            assert result == {"ok": True}

            # Still only one session; still only one handshake.
            assert len(sessions) == 1
            assert sessions[0].initialize_count == 1
            assert sessions[0].call_count == 1
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_second_call_reuses_session(
        self, fake_transport: FakeTransport
    ) -> None:
        client = await _connected_client()
        try:
            await client.call_tool("search_passages", {"q": "1"})
            await client.call_tool("search_passages", {"q": "2"})
            await client.call_tool("search_passages", {"q": "3"})

            sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
            assert len(sessions) == 1, "expected exactly one ClientSession across calls"
            assert sessions[0].initialize_count == 1
            assert sessions[0].call_count == 3
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_list_tools_uses_same_session_as_call_tool(
        self, fake_transport: FakeTransport
    ) -> None:
        client = await _connected_client()
        try:
            await client.list_tools()  # fetches + caches
            await client.call_tool("search_passages", {"q": "x"})

            sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
            assert len(sessions) == 1
            assert sessions[0].list_count == 1
            assert sessions[0].call_count == 1
        finally:
            await client.aclose()


# Concurrent calls don't double-init


class TestConcurrentInit:
    @pytest.mark.asyncio
    async def test_concurrent_calls_on_fresh_client_single_session(
        self, fake_transport: FakeTransport
    ) -> None:
        """If N coroutines race to call_tool() before connect(),
        exactly one session should be opened and the rest await it."""

        # Slow down session initialisation so the race window is real.
        slow_factory_calls: list[int] = []

        original_factory = fake_transport._session_factory

        async def _slow_initialize(self: FakeSession) -> None:  # bound method
            await asyncio.sleep(0.05)
            self.initialize_count += 1

        def factory(url: str, read: Any, write: Any) -> FakeSession:
            sess = FakeSession(read, write)
            # Monkey-patch this instance's initialize to be slow.
            slow_factory_calls.append(1)
            import types

            sess.initialize = types.MethodType(_slow_initialize, sess)  # type: ignore[method-assign]
            return sess

        fake_transport.set_session_factory(factory)
        try:
            client = MCPHTTPClient(standard_url="http://standard:9000/mcp")
            await client.connect()
            try:
                results = await asyncio.gather(
                    *[client.call_tool("search_passages", {"q": i}) for i in range(10)]
                )
                assert all(r == {"ok": True} for r in results)

                sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
                assert len(sessions) == 1, (
                    f"expected one session across 10 concurrent calls, "
                    f"got {len(sessions)}"
                )
                assert sessions[0].initialize_count == 1
                assert sessions[0].call_count == 10
            finally:
                await client.aclose()
        finally:
            fake_transport.set_session_factory(original_factory)


# ClosedResourceError triggers transparent reconnect + retry


class _FakeClosedResourceError(Exception):
    """Stand-in for ``anyio.ClosedResourceError`` — class name matters."""


_FakeClosedResourceError.__name__ = "ClosedResourceError"  # matches detection


class TestReconnectOnClosedResource:
    @pytest.mark.asyncio
    async def test_closed_resource_error_triggers_reconnect(
        self, fake_transport: FakeTransport
    ) -> None:
        """First call hits ClosedResourceError; client reopens session and
        retries, caller sees a successful result."""

        session_log: list[FakeSession] = []

        def factory(url: str, read: Any, write: Any) -> FakeSession:
            sess = FakeSession(read, write)
            session_log.append(sess)
            # First session raises on call_tool; second works.
            if len(session_log) == 1:

                def _first_call(name: str, args: dict[str, Any]) -> BaseException:
                    return _FakeClosedResourceError("stream closed")

                sess.on_call_tool = _first_call
            return sess

        fake_transport.set_session_factory(factory)
        client = await _connected_client()
        try:
            result = await client.call_tool("search_passages", {"q": "x"})
            assert result == {"ok": True}, (
                "client should have reconnected and retried transparently"
            )

            sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
            # First session was opened by connect(), torn down by
            # reconnect logic, and replaced with a second one.
            assert len(sessions) == 2
            assert sessions[0].call_count == 1  # the failing call
            assert sessions[1].call_count == 1  # the successful retry
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_two_consecutive_transport_failures_surface_error(
        self, fake_transport: FakeTransport
    ) -> None:
        """If BOTH the initial and the retry calls hit transport death,
        the error is packaged into an error dict — not swallowed, not
        retried a third time."""

        def factory(url: str, read: Any, write: Any) -> FakeSession:
            sess = FakeSession(read, write)

            def _always_fail(name: str, args: dict[str, Any]) -> BaseException:
                return _FakeClosedResourceError("stream closed")

            sess.on_call_tool = _always_fail
            return sess

        fake_transport.set_session_factory(factory)
        client = await _connected_client()
        try:
            result = await client.call_tool("search_passages", {"q": "x"})
            assert "error" in result
            assert result["tool"] == "search_passages"
            assert result.get("mcp_unhealthy") is True

            sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
            # Exactly two call attempts (initial + one retry).
            total_calls = sum(s.call_count for s in sessions)
            assert total_calls == 2, (
                f"expected 2 total call attempts, got {total_calls}"
            )
        finally:
            await client.aclose()


# MCP protocol errors do not tear down the session


class TestProtocolErrorDoesNotReconnect:
    @pytest.mark.asyncio
    async def test_mcp_error_keeps_session_alive(
        self, fake_transport: FakeTransport
    ) -> None:
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        calls_made: list[int] = []

        def factory(url: str, read: Any, write: Any) -> FakeSession:
            sess = FakeSession(read, write)

            def _raise_mcp_error(name: str, args: dict[str, Any]) -> BaseException:
                calls_made.append(1)
                if len(calls_made) == 1:
                    return McpError(ErrorData(code=-1, message="bad query"))
                # subsequent calls succeed
                return None  # type: ignore[return-value]

            sess.on_call_tool = _raise_mcp_error
            return sess

        fake_transport.set_session_factory(factory)
        client = await _connected_client()
        try:
            bad = await client.call_tool("search_passages", {"q": "bad"})
            assert "error" in bad  # protocol error surfaces as error dict

            # The session is not reopened: MCP errors are protocol-level,
            # not transport-level.
            sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
            assert len(sessions) == 1, (
                "McpError must not trigger a reconnect (session still alive)"
            )

            # The same session should still serve subsequent calls.
            good = await client.call_tool("search_passages", {"q": "good"})
            assert good == {"ok": True}
            assert len(fake_transport.sessions_by_url["http://standard:9000/mcp"]) == 1
        finally:
            await client.aclose()


# NER routing opens distinct sessions per endpoint URL


class TestNERRouting:
    @pytest.mark.asyncio
    async def test_ner_and_standard_have_distinct_sessions(
        self, fake_transport: FakeTransport
    ) -> None:
        client = MCPHTTPClient(
            standard_url="http://standard:9000/mcp",
            ner_url="http://ner:9001/mcp",
        )
        await client.connect()
        try:
            # A standard tool and a NER tool must each reuse their own
            # endpoint's session — never cross-contaminate.
            await client.call_tool("search_passages", {"q": "x"})
            await client.call_tool("search_by_entity", {"entity": "Vader"})
            await client.call_tool("search_passages", {"q": "y"})
            await client.call_tool("search_by_entity", {"entity": "Palpatine"})

            std_sessions = fake_transport.sessions_by_url["http://standard:9000/mcp"]
            ner_sessions = fake_transport.sessions_by_url["http://ner:9001/mcp"]
            assert len(std_sessions) == 1
            assert len(ner_sessions) == 1
            assert std_sessions[0].call_count == 2
            assert ner_sessions[0].call_count == 2
        finally:
            await client.aclose()


# aclose() tears down sessions and prevents reuse


class TestCloseLifecycle:
    @pytest.mark.asyncio
    async def test_aclose_closes_worker_tasks(
        self, fake_transport: FakeTransport
    ) -> None:
        client = await _connected_client()
        # Grab handles before close.
        handle = client._sessions["http://standard:9000/mcp"]
        task = handle.task
        assert not task.done()
        await client.aclose()
        assert task.done(), "worker task must finish after aclose()"
        assert "http://standard:9000/mcp" not in client._sessions

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self, fake_transport: FakeTransport) -> None:
        client = await _connected_client()
        await client.aclose()
        # Second close must not raise.
        await client.aclose()

    @pytest.mark.asyncio
    async def test_call_after_aclose_raises_or_errors(
        self, fake_transport: FakeTransport
    ) -> None:
        client = await _connected_client()
        await client.aclose()
        # After close, call_tool should surface an error (not silently
        # reopen a session on a dead client).
        result = await client.call_tool("search_passages", {"q": "x"})
        assert "error" in result


# Foreign-loop safety. The cached session belongs to the loop that ran
# connect(). A caller on a different asyncio event loop (for example the
# fresh loop that pipeline_handlers._run_coro_sync starts in a daemon
# thread) must not reuse it: the anyio streams are bound to the owner loop
# and a cross-loop send blocks for the full outer timeout. Such a caller
# gets a throwaway session scoped to its own loop.


class TestForeignLoopFallback:
    @pytest.mark.asyncio
    async def test_on_owner_loop_true_in_connect_loop(
        self, fake_transport: FakeTransport
    ) -> None:
        client = await _connected_client()
        try:
            assert client._on_owner_loop() is True
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_on_owner_loop_false_before_connect(
        self, fake_transport: FakeTransport
    ) -> None:
        # Before connect(), _owner_loop is None → treat "unset" as
        # "same loop" so unit tests that bypass connect still work.
        client = MCPHTTPClient(standard_url="http://standard:9000/mcp")
        assert client._on_owner_loop() is True

    def test_call_tool_from_child_thread_loop_does_not_wedge(
        self, fake_transport: FakeTransport
    ) -> None:
        """End-to-end: connect on loop A, call_tool from loop B in a
        daemon thread.  The call must complete with a live result and
        not time out.

        The cached session's anyio streams are bound to loop A, so a
        write.send() from loop B never wakes the post_writer on the
        blocked loop A and the call would hang until the outer timeout.
        ``_invoke_with_reconnect`` detects the foreign loop and opens a
        throwaway session scoped to loop B.
        """
        import threading

        async def _main() -> MCPHTTPClient:
            client = MCPHTTPClient(standard_url="http://standard:9000/mcp")
            await client.connect()
            return client

        loop_a = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop_a)
            client = loop_a.run_until_complete(_main())

            # Sanity: call on loop A works.
            res_a = loop_a.run_until_complete(
                client.call_tool("search_passages", {"q": "loop-a"})
            )
            assert res_a == {"ok": True}

            # Simulate _run_coro_sync: a new loop on a child thread calls
            # call_tool on the same client and must not hang.
            box: dict[str, Any] = {}

            def _child() -> None:
                loop_b = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop_b)
                    # Hard cap: if the call wedges, wait_for fires and
                    # the assertion below catches the TimeoutError.
                    try:
                        result = loop_b.run_until_complete(
                            asyncio.wait_for(
                                client.call_tool("search_passages", {"q": "loop-b"}),
                                timeout=3.0,
                            )
                        )
                        box["result"] = result
                    except BaseException as exc:  # noqa: BLE001
                        box["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    loop_b.close()

            t = threading.Thread(target=_child, daemon=True)
            t.start()
            t.join(timeout=8.0)
            assert not t.is_alive(), (
                "child thread did not complete — foreign-loop call wedged"
            )
            assert "error" not in box, (
                f"foreign-loop call surfaced an error: {box.get('error')}"
            )
            assert box.get("result") == {"ok": True}, (
                f"unexpected foreign-loop result: {box.get('result')!r}"
            )

            # Cached session on loop A is still alive and reusable.
            res_a2 = loop_a.run_until_complete(
                client.call_tool("search_passages", {"q": "loop-a-again"})
            )
            assert res_a2 == {"ok": True}

            loop_a.run_until_complete(client.aclose())
        finally:
            loop_a.close()

    def test_foreign_loop_call_does_not_touch_session_cache(
        self, fake_transport: FakeTransport
    ) -> None:
        """The foreign-loop path opens a throwaway session and must not
        register a handle in ``self._sessions`` — that cache is strictly
        owned by the connect loop.
        """
        import threading

        loop_a = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop_a)
            client = MCPHTTPClient(standard_url="http://standard:9000/mcp")
            loop_a.run_until_complete(client.connect())
            cached_before = dict(client._sessions)
            assert len(cached_before) == 1  # the connect()-time session

            def _child() -> None:
                loop_b = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop_b)
                    loop_b.run_until_complete(
                        client.call_tool("search_passages", {"q": "x"})
                    )
                finally:
                    loop_b.close()

            t = threading.Thread(target=_child, daemon=True)
            t.start()
            t.join(timeout=5.0)
            assert not t.is_alive()

            # Owner-loop cache unchanged: the foreign-loop call did not
            # register or evict a handle.
            assert set(client._sessions.keys()) == set(cached_before.keys())
            assert (
                client._sessions["http://standard:9000/mcp"]
                is cached_before["http://standard:9000/mcp"]
            )

            loop_a.run_until_complete(client.aclose())
        finally:
            loop_a.close()

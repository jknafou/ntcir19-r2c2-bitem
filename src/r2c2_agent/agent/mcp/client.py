"""MCP client using the official MCP SDK with Streamable HTTP transport.

Uses ``mcp.ClientSession`` over ``streamablehttp_client`` with session
reuse across calls.  A long-lived session is held open in a dedicated
background task per endpoint; all ``call_tool`` / ``list_tools``
invocations share that session and multiplex over its streams.

Session reuse
-------------
When ``call_tool`` is invoked in parallel, opening a fresh session per
call produces frequent ``anyio.ClosedResourceError`` and "unhandled
errors in a TaskGroup" failures, because the tear-down of one session
races the set-up of the next.  The MCP protocol is id-multiplexed and
the SDK's ``ClientSession`` is safe to use from multiple coroutines, so
one shared session avoids the ``async with`` churn.

Threading / concurrency model
-----------------------------
* First call: the initializing coroutine holds the per-URL
  ``asyncio.Lock`` while a detached ``asyncio.Task`` opens the
  transport + session and parks on a shutdown event.  Concurrent
  callers on a fresh client await the same lock, so only one task
  performs the handshake.
* Steady state: callers share the same ``ClientSession`` object and
  issue ``session.call_tool(...)`` directly.  The session multiplexes.
* Transport death: if a call raises a closed / broken resource error,
  the client tears the dead session down, reconnects once, and
  retries the call once.  A second consecutive failure is returned
  as an error dict.

The detached worker task is what makes this safe on Python 3.14 +
anyio: both the open and close of the ``AsyncExitStack`` happen inside
one task's own cancel scope, never straddling a caller's TaskGroup.

Routing
-------
NER-specific tools are routed to the NER MCP endpoint using the
explicit ``NER_TOOLS`` set from config; everything else goes to the
standard endpoint.  Each endpoint has its own reusable session.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from r2c2_agent.agent.config import NER_TOOLS
from r2c2_agent.agent.exceptions import MCPError

__all__ = ["MCPHTTPClient", "mcp_tool_to_openai"]

log = logging.getLogger(__name__)


# Substrings that indicate the transport (not the protocol) is dead and
# the session must be reopened.  Matched against ``str(root_cause).lower()``.
_UNHEALTHY_NEEDLES: tuple[str, ...] = (
    "closedresourceerror",
    "brokenresourceerror",
    "session is closed",
    "connection closed",
    "connection reset",
    "connection refused",
    "connection aborted",
    "server disconnected",
    "task group",  # "unhandled errors in a TaskGroup"
    "taskgroup",
)


def mcp_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an MCP tool schema to OpenAI function-calling format.

    Args:
        tool: MCP tool definition with ``name``, ``description``,
            and ``inputSchema`` keys.

    Returns:
        OpenAI-format tool definition.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
        },
    }


def _walk_root_cause(exc: BaseException, max_depth: int = 8) -> BaseException:
    """Walk ``__cause__`` / ``__context__`` to find the root exception.

    ``TaskGroup`` and the MCP SDK bury the real transport error deep in
    the chain; callers need the leaf to decide "unhealthy vs. not".
    """
    root: BaseException = exc
    for _ in range(max_depth):
        inner = root.__cause__ or root.__context__
        if inner is None or inner is root:
            break
        root = inner
    return root


def _is_transport_unhealthy(exc: BaseException) -> bool:
    """Return True if ``exc`` (or its root cause) looks like transport death."""
    # Check the exception class name (handles anyio.ClosedResourceError even
    # when str() is empty, which it often is).
    for cur in (exc, _walk_root_cause(exc)):
        cls_name = type(cur).__name__.lower()
        if "closedresource" in cls_name or "brokenresource" in cls_name:
            return True
    # Fall back to string matching on both the top-level message and the
    # root-cause message — anyio wraps closed streams in TaskGroup errors.
    haystacks = [str(exc).lower(), str(_walk_root_cause(exc)).lower()]
    return any(needle in hay for hay in haystacks for needle in _UNHEALTHY_NEEDLES)


@dataclass
class _SessionHandle:
    """Per-URL session + the task keeping it alive.

    ``task`` is a detached ``asyncio.Task`` that owns the ``async with``
    scope for both ``streamablehttp_client`` and ``ClientSession``.  It
    parks on ``shutdown`` until the client is closed or the session
    dies, at which point the ``async with`` stacks unwind inside the
    task's own cancel scope — avoiding the cross-task teardown race.
    """

    session: ClientSession
    task: asyncio.Task[None]
    shutdown: asyncio.Event
    dead: bool = False
    error: BaseException | None = field(default=None)


class MCPHTTPClient:
    """MCP client with per-endpoint session reuse and one-shot reconnect.

    Each endpoint URL (standard and, optionally, NER) gets at most one
    live ``ClientSession``.  The session is opened lazily on the first
    call that needs it and reused for every subsequent call.  On
    detected transport death the session is torn down and reopened once
    before the call is retried; a second failure surfaces as an error
    dict.

    Args:
        standard_url: URL for the standard MCP server endpoint
            (e.g. ``"http://host:9000/mcp"``).
        ner_url: URL for the NER-enriched MCP server endpoint.
            If empty, NER tools are sent to the standard server.
        timeout: Default timeout in seconds for tool calls.

    Example::

        async with MCPHTTPClient("http://host:9000/mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_passages", {"query": "test"})
    """

    def __init__(
        self,
        standard_url: str,
        ner_url: str = "",
        timeout: float = 30.0,
    ) -> None:
        self._standard_url = standard_url.rstrip("/")
        self._ner_url = ner_url.rstrip("/") if ner_url else self._standard_url
        self._timeout = timeout
        self._tools_cache: list[dict[str, Any]] | None = None
        self._connected = False
        self._ner_available = False

        # Per-URL session handles and init locks.  ``asyncio.Lock`` is
        # lazy w.r.t. the event loop in Python 3.10+ — safe to construct
        # before any loop is running.
        self._sessions: dict[str, _SessionHandle] = {}
        self._init_locks: dict[str, asyncio.Lock] = {}
        self._closed: bool = False

        # Event loop that owns the cached sessions.  Captured at
        # ``connect()`` time.  Callers on a different loop (e.g. a
        # fresh loop running in a child thread) must not touch the cached
        # session: its anyio memory streams are bound to this loop and
        # cross-loop sends wedge until the owner loop becomes idle.
        # See ``_invoke_adhoc`` for the fallback path.
        self._owner_loop: asyncio.AbstractEventLoop | None = None

    # Connection lifecycle

    async def connect(self) -> None:
        """Verify connectivity by eagerly opening the standard session.

        Opens (and keeps open) a session against the standard endpoint;
        if an NER endpoint is configured, also opens a session against
        it.  NER failures are non-fatal.

        Idempotent: calling when already connected is a no-op.

        Raises:
            MCPError: If connection to the standard server fails.
        """
        if self._connected:
            return
        if self._closed:
            raise MCPError("MCPHTTPClient has been closed; create a new instance.")

        # Capture the owner loop so ``_on_owner_loop`` can distinguish the
        # main-loop fast path from foreign-loop ad-hoc fallbacks.
        self._owner_loop = asyncio.get_running_loop()

        # Open (and hold open) the standard session.  Failure here is fatal.
        await self._open_session_with_retry(
            self._standard_url,
            retries=3,
            backoff=5.0,
        )

        # Open the NER session if distinct.  Failure here is non-fatal
        # — NER tools will transparently fall back to the standard URL.
        if self._ner_url != self._standard_url:
            try:
                await self._open_session_with_retry(
                    self._ner_url, retries=3, backoff=5.0
                )
                self._ner_available = True
            except MCPError:
                log.warning(
                    "Failed to connect to NER endpoint %s, using standard",
                    self._ner_url,
                )
                self._ner_available = False
        else:
            self._ner_available = True  # same endpoint

        self._connected = True

    async def _open_session_with_retry(
        self,
        url: str,
        retries: int = 3,
        backoff: float = 5.0,
    ) -> _SessionHandle:
        """Open a reusable session with exponential-backoff retries.

        Retries handle MCP servers that report readiness before their
        backend (Elasticsearch, FAISS, KG) is fully loaded.

        Raises:
            MCPError: If all attempts fail.
        """
        last_exc: BaseException | None = None
        for attempt in range(1 + retries):
            try:
                handle = await self._open_session(url)
                log.info("Connectivity OK: %s", url)
                return handle
            except Exception as exc:  # noqa: BLE001 — retry-all is the contract
                last_exc = exc
                if attempt < retries:
                    wait = backoff * (2**attempt)
                    log.warning(
                        "MCP connect attempt %d/%d failed (%s): %s — retry in %.0fs",
                        attempt + 1,
                        1 + retries,
                        url,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)

        raise MCPError(
            f"Failed to connect to MCP at {url} after {1 + retries} attempts: "
            f"{last_exc}"
        ) from last_exc

    async def _open_session(self, url: str) -> _SessionHandle:
        """Open a fresh ``ClientSession`` in a detached worker task.

        Blocks until the worker has either finished the MCP handshake
        (``session.initialize()``) or errored out.  Subsequent callers
        may use ``handle.session`` directly.
        """
        ready = asyncio.Event()
        shutdown = asyncio.Event()
        slot: dict[str, Any] = {}

        async def _worker() -> None:
            """Hold the transport + session open until ``shutdown`` fires."""
            try:
                async with streamablehttp_client(url) as transport:
                    # streamablehttp_client yields (read, write, get_session_id).
                    read, write, *_ = transport
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        slot["session"] = session
                        ready.set()
                        # Park here; teardown happens when shutdown fires.
                        await shutdown.wait()
            except BaseException as exc:  # noqa: BLE001 — propagate to opener
                slot["error"] = exc
                # Make sure a caller awaiting ``ready`` is unblocked.
                ready.set()
                # Other errors reach the opener through ``slot``; only
                # cancellation is re-raised, so the task ends as cancelled.
                if isinstance(exc, asyncio.CancelledError):
                    raise

        task = asyncio.create_task(_worker(), name=f"mcp-session:{url}")
        await ready.wait()

        if "error" in slot:
            # Handshake failed.  Ensure the worker task finishes cleanly
            # before raising.
            shutdown.set()
            try:
                await task
            except BaseException:  # noqa: BLE001 — best-effort cleanup
                pass
            err = slot["error"]
            raise MCPError(f"Failed to open MCP session to {url}: {err}") from err

        handle = _SessionHandle(session=slot["session"], task=task, shutdown=shutdown)
        self._sessions[url] = handle
        return handle

    async def _close_session_handle(self, url: str) -> None:
        """Tear down the session held for ``url`` (if any).

        Signals the worker's shutdown event and awaits task completion
        so the ``async with`` stacks unwind inside the worker's cancel
        scope.  Safe to call even if no session was ever opened for
        this URL.
        """
        handle = self._sessions.pop(url, None)
        if handle is None:
            return
        handle.dead = True
        handle.shutdown.set()
        try:
            # Give the worker a short grace period to unwind cleanly.
            await asyncio.wait_for(handle.task, timeout=10.0)
        except TimeoutError:
            log.warning(
                "MCP session worker for %s did not exit in 10s; cancelling", url
            )
            handle.task.cancel()
            try:
                await handle.task
            except (asyncio.CancelledError, BaseException):  # noqa: BLE001
                pass
        except BaseException as exc:  # noqa: BLE001 — best-effort cleanup
            log.debug("MCP session worker for %s exited with: %s", url, exc)

    async def close(self) -> None:
        """Close all live sessions and reset client state.

        After ``close()``, the client cannot be re-opened — create a
        new instance instead.  Idempotent.
        """
        if self._closed:
            return
        for url in list(self._sessions.keys()):
            await self._close_session_handle(url)
        self._connected = False
        self._tools_cache = None
        self._closed = True

    # ``aclose`` is the conventional name for an async close.
    aclose = close

    async def __aenter__(self) -> MCPHTTPClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # Routing + session acquisition

    def _ensure_connected(self) -> None:
        """Raise if not connected.

        Raises:
            MCPError: If ``connect()`` has not been called.
        """
        if not self._connected:
            raise MCPError(
                "MCPHTTPClient not connected. Call connect() or use 'async with'."
            )

    def _url_for_tool(self, name: str) -> str:
        """Return the endpoint URL to use for a given tool name.

        Args:
            name: Tool name.

        Returns:
            The NER URL for NER tools (if available), standard URL otherwise.
        """
        if name in NER_TOOLS and self._ner_available:
            return self._ner_url
        return self._standard_url

    def _lock_for(self, url: str) -> asyncio.Lock:
        """Return (creating if necessary) the init lock for a URL."""
        lock = self._init_locks.get(url)
        if lock is None:
            lock = asyncio.Lock()
            self._init_locks[url] = lock
        return lock

    def _on_owner_loop(self) -> bool:
        """Return True iff the current running loop owns the cached sessions.

        The cached ``ClientSession`` and the anyio memory object streams
        it wraps are bound to ``self._owner_loop`` (captured in
        ``connect()``).  Reusing them from a different asyncio event loop
        (e.g. a new loop running inside a daemon thread) wedges:
        ``write_stream.send(...)`` queues a callback that only
        the owner loop can run, but the owner loop is blocked on
        ``thread.join(...)`` waiting for this very call to finish.

        When this returns False the caller must route through
        ``_invoke_adhoc``, which opens a transient
        ``streamablehttp_client`` + ``ClientSession`` scoped to the
        current loop.
        """
        if self._owner_loop is None:
            # ``connect()`` hasn't run (unit tests that bypass connect
            # via mocks, etc.).  Treat as "on owner" so those callers
            # use the cached session.
            return True
        try:
            cur = asyncio.get_running_loop()
        except RuntimeError:
            return False
        return cur is self._owner_loop

    async def _invoke_adhoc(
        self, url: str, name: str, arguments: dict[str, Any]
    ) -> Any:
        """Open a throwaway session on the current loop, call the tool, close.

        One session per call, for foreign-loop callers.  Does not touch
        ``self._sessions`` (that cache belongs to the owner loop).  The
        returned value is the raw ``CallToolResult``
        SDK object, matching ``session.call_tool`` so ``call_tool`` can
        post-process it uniformly.
        """
        async with streamablehttp_client(url) as transport:
            read, write, *_ = transport
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    async def _list_tools_adhoc(self, url: str) -> Any:
        """Foreign-loop counterpart of ``_fetch_tools`` — fresh session."""
        async with streamablehttp_client(url) as transport:
            read, write, *_ = transport
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.list_tools()

    async def _get_session(self, url: str) -> ClientSession:
        """Acquire a live session for ``url``, (re)opening if needed.

        Holds the per-URL lock across the entire (re)initialization so
        two concurrent callers on a fresh client never double-init.
        Steady-state lookups are lock-free (the lock is only taken on a
        miss or when the session is dead).
        """
        if self._closed:
            raise MCPError("MCPHTTPClient has been closed.")

        # Fast path: live, healthy session already cached.
        handle = self._sessions.get(url)
        if handle is not None and not handle.dead:
            return handle.session

        async with self._lock_for(url):
            handle = self._sessions.get(url)
            if handle is not None and not handle.dead:
                return handle.session
            # Miss or dead — (re)open.  ``_open_session`` registers the
            # new handle in ``self._sessions``.
            new_handle = await self._open_session(url)
            return new_handle.session

    # Public API

    async def list_tools(self) -> list[dict[str, Any]]:
        """Fetch tool schemas and convert to OpenAI format.

        Uses the reusable session; result is cached after the first call.

        Returns:
            Tool definitions in OpenAI function-calling format.

        Raises:
            MCPError: On communication failure or if not connected.
        """
        if self._tools_cache is not None:
            return self._tools_cache

        self._ensure_connected()

        try:
            mcp_tools = await self._fetch_tools(self._standard_url)

            # Also fetch NER-only tools if NER endpoint differs
            if self._ner_available and self._ner_url != self._standard_url:
                try:
                    ner_tools = await self._fetch_tools(self._ner_url)
                    standard_names = {t["name"] for t in mcp_tools}
                    for t in ner_tools:
                        if t["name"] not in standard_names:
                            mcp_tools.append(t)
                except MCPError:
                    log.warning("Failed to fetch NER tools, continuing without")

            openai_tools = [mcp_tool_to_openai(t) for t in mcp_tools]
            self._tools_cache = openai_tools
            return openai_tools

        except McpError as exc:
            raise MCPError(f"Failed to list tools: {exc}") from exc

    async def _fetch_tools(self, url: str) -> list[dict[str, Any]]:
        """List tools via the reusable session for ``url``.

        On detected transport death, tears the session down, reopens
        once, and retries once before surfacing the error.

        Foreign-loop callers (see ``_on_owner_loop``) bypass the cache
        and use a per-call session, avoiding the cross-loop wedge.
        """
        if not self._on_owner_loop():
            try:
                result = await self._list_tools_adhoc(url)
            except Exception as exc:  # noqa: BLE001
                raise MCPError(
                    f"Failed to list tools from {url} (foreign-loop): {exc}"
                ) from exc
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema,
                }
                for t in result.tools
            ]

        try:
            session = await self._get_session(url)
            result = await session.list_tools()
        except Exception as exc:  # noqa: BLE001 — classify then decide
            if _is_transport_unhealthy(exc):
                log.warning(
                    "list_tools(%s) transport unhealthy (%s); reconnecting once",
                    url,
                    type(exc).__name__,
                )
                await self._close_session_handle(url)
                try:
                    session = await self._get_session(url)
                    result = await session.list_tools()
                except Exception as exc2:  # noqa: BLE001
                    raise MCPError(
                        f"Failed to list tools from {url} after reconnect: {exc2}"
                    ) from exc2
            else:
                raise MCPError(f"Failed to list tools from {url}: {exc}") from exc

        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Invoke an MCP tool and return the result.

        Uses the reusable session for the relevant endpoint.  On
        detected transport death, reconnects once and retries once
        before surfacing the error.

        Args:
            name: Tool name (e.g. ``"search_passages"``).
            arguments: Tool arguments.
            timeout: Per-call timeout in seconds (reserved for future use).

        Returns:
            Tool result as a dict.  On error, contains an ``"error"`` key
            (and ``"mcp_unhealthy": True`` when the underlying transport
            was diagnosed as dead).
        """
        url = self._url_for_tool(name)

        try:
            result, unhealthy_after = await self._invoke_with_reconnect(
                url, name, arguments
            )
        except MCPError as exc:
            log.warning("Tool %s failed: %s", name, exc)
            return {"error": str(exc), "tool": name}
        except McpError as exc:
            # Protocol error from the SDK — session is still alive.
            log.warning("Tool %s failed (SDK): %s", name, exc)
            return {"error": str(exc), "tool": name}
        except Exception as exc:  # noqa: BLE001 — surface unhealthy hint
            root = _walk_root_cause(exc)
            unhealthy = _is_transport_unhealthy(exc)
            log.warning(
                "Tool %s failed: %s (unhealthy_transport=%s)",
                name,
                exc,
                unhealthy,
            )
            out: dict[str, Any] = {
                "error": str(exc),
                "tool": name,
                "error_class": type(root).__name__,
            }
            if unhealthy:
                out["mcp_unhealthy"] = True
            return out

        # Extract text content blocks from the SDK result
        texts: list[str] = []
        for block in result.content:
            if isinstance(block, TextContent):
                texts.append(block.text)

        # Try to parse the first text block as JSON (most tools return JSON)
        if texts:
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, IndexError):
                return {"text": "\n".join(texts)}

        # No text content -- check for error
        if result.isError:
            out = {"error": f"Tool {name} returned error with no text content"}
            if unhealthy_after:
                out["mcp_unhealthy"] = True
            return out
        return {"result": "empty"}

    async def _invoke_with_reconnect(
        self,
        url: str,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[Any, bool]:
        """Invoke a tool, reconnecting once on transport death.

        Returns ``(result, unhealthy_after_retry)``.  The second element
        is True only when even the retry observed an unhealthy transport
        but did not raise — reserved for future use.

        Raises:
            McpError: Protocol error from the SDK (session alive).
            Exception: Any non-transport error, or the retry's error.
        """
        # Foreign-loop fallback: the cached session is bound to
        # ``self._owner_loop``; reusing its anyio streams from a
        # different loop (e.g. a loop running in a child thread)
        # wedges for the full outer timeout because the send-side
        # callback can only be drained by the blocked owner loop.  Open
        # a throwaway session on the current loop and make the call
        # there.  No reconnect logic is needed — the session lives for
        # exactly one call and is torn down on exit.
        if not self._on_owner_loop():
            return await self._invoke_adhoc(url, name, arguments), False

        session = await self._get_session(url)
        try:
            return await session.call_tool(name, arguments), False
        except McpError:
            # Protocol error — session is still alive; do not reconnect.
            raise
        except Exception as exc:  # noqa: BLE001 — decide reconnect
            if not _is_transport_unhealthy(exc):
                raise
            log.warning(
                "call_tool(%s) transport unhealthy (%s); reconnecting once",
                name,
                type(exc).__name__,
            )
            await self._close_session_handle(url)
            session = await self._get_session(url)
            # Retry once.  If this also fails, let the exception propagate
            # to the caller for error-dict packaging.
            return await session.call_tool(name, arguments), False

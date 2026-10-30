"""Protocol definition for MCP clients."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["MCPClientProtocol"]


@runtime_checkable
class MCPClientProtocol(Protocol):
    """Interface for MCP tool-calling clients."""

    async def list_tools(self) -> list[dict[str, Any]]:
        """Fetch available tool schemas from the MCP server.

        Returns:
            List of tool definitions in OpenAI function-calling format.
        """
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Invoke an MCP tool and return the result.

        Args:
            name: Tool name (e.g. ``"search_passages"``).
            arguments: Tool arguments as a dict.
            timeout: Maximum seconds to wait for the MCP response.

        Returns:
            Tool result as a dict. On error, contains an ``"error"`` key.
        """
        ...

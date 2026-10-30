"""Protocol definition for LLM clients."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from r2c2_agent.agent.interfaces.types import LLMResponse

__all__ = ["LLMClientProtocol"]


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Interface for LLM clients (vLLM, Ollama, dummy)."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
    ) -> LLMResponse:
        """Send a chat completion request and return a parsed response.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Tool definitions in OpenAI function-calling format.
                Pass ``None`` or ``[]`` to disable tool calling.
            enable_thinking: Whether to request thinking/reasoning tokens.

        Returns:
            Parsed LLMResponse with content, reasoning, and tool_calls.
        """
        ...

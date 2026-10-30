"""Custom exception hierarchy for Agent-BiTeM.

All exceptions inherit from ``AgentBiTemError`` so callers can catch
the entire family with a single handler when needed.
"""

__all__ = [
    "AgentBiTemError",
    "BudgetExceededError",
    "LLMError",
    "MCPError",
    "ParseError",
    "ToolDispatchError",
    "ValidationError",
]


class AgentBiTemError(Exception):
    """Base exception for all Agent-BiTeM errors."""


class LLMError(AgentBiTemError):
    """Error communicating with the LLM (vLLM / Ollama)."""


class MCPError(AgentBiTemError):
    """Error communicating with an MCP server."""


class ToolDispatchError(AgentBiTemError):
    """Error dispatching or sanitizing a tool call."""


class ParseError(AgentBiTemError):
    """Error parsing structured output (verifier response, nuggets, etc.)."""


class BudgetExceededError(AgentBiTemError):
    """Context token budget exceeded after compaction attempts."""


class ValidationError(AgentBiTemError):
    """PR/AC submission validation failure."""

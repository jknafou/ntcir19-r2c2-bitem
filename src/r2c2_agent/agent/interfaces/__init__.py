"""Interface protocols and shared types for Agent-BiTeM."""

from r2c2_agent.agent.interfaces.llm import LLMClientProtocol
from r2c2_agent.agent.interfaces.mcp import MCPClientProtocol
from r2c2_agent.agent.interfaces.types import (
    ACResult,
    CitedPassage,
    LLMResponse,
    Nugget,
    PREntry,
    Question,
    QuestionTrace,
    ToolCall,
    ToolResult,
    VerificationResult,
)

__all__ = [
    "ACResult",
    "CitedPassage",
    "LLMClientProtocol",
    "LLMResponse",
    "MCPClientProtocol",
    "Nugget",
    "PREntry",
    "Question",
    "QuestionTrace",
    "ToolCall",
    "ToolResult",
    "VerificationResult",
]

"""OpenAI-compatible LLM client for vLLM and Ollama.

Wraps the ``openai.AsyncOpenAI`` SDK to handle Qwen3's reasoning
tokens (via vLLM's ``--reasoning-parser qwen3``) and parallel tool
calls.  Non-streaming.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from r2c2_agent.agent.exceptions import LLMError
from r2c2_agent.agent.interfaces.types import LLMResponse, ToolCall

__all__ = ["OpenAILLMClient"]

log = logging.getLogger(__name__)


class OpenAILLMClient:
    """LLM client backed by an OpenAI-compatible HTTP endpoint.

    Args:
        base_url: vLLM or Ollama endpoint (e.g. ``"http://host:8000/v1"``).
        model: Model identifier (e.g. ``"Qwen/Qwen3.5-122B-A10B-FP8"``).
        api_key: API key.  Defaults to ``"EMPTY"`` (vLLM convention).
        temperature: Sampling temperature.  ``None`` uses the server default
            (typically 0 for greedy).  Set to e.g. ``0.7`` for RL rollouts.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float | None = None,
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
        user: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        guided_regex: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat completion and return a parsed ``LLMResponse``.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Tool definitions in OpenAI function-calling format.
            enable_thinking: Whether to request reasoning tokens.
            user: Optional opaque identifier passed through to vLLM's
                OpenAI-compatible ``user`` field. The driver packs
                ``"qid=<id>;iter=<n>;phase=<REASON|WRITE>"`` so vLLM access
                logs can be grepped by qid. Ignored when empty/None.
            max_tokens: Per-call token ceiling. When ``None`` the server
                default applies. Used by the finalizer ladder to cap
                tier-2+ guess generations at ``FINALIZER_MAX_TOKENS``.
            temperature: Per-call sampling temperature. When ``None`` the
                client's constructor default applies. Used by the
                finalizer ladder (``FINALIZER_TEMPERATURE``) and by
                ensemble rollouts.
            guided_regex: Optional regex pattern forwarded to vLLM's
                structured-output hook so the server constrains the
                output to match. Enforces the
                ``Answer: <x> | UNKNOWN`` schema on Tier-2+ emissions
                without relying on prompt compliance.

                Transport: vLLM 0.17 takes the pattern as
                ``structured_outputs.regex`` inside the request body
                and ignores a top-level ``guided_regex`` field with
                only a ``The following fields were present in the
                request but ignored: {'guided_regex'}`` warning, after
                which decoding is unconstrained. The kwarg is therefore
                sent as
                ``extra_body["structured_outputs"] = {"regex": ...}``.
                Its Python name is kept so call sites and tests do not
                depend on the vLLM version.
            tool_choice: Optional explicit tool_choice to forward to vLLM.
                When set, overrides the default ``"auto"`` used when
                ``tools`` is passed. Use ``"required"`` to force the model
                to emit exactly one tool call (OpenAI spec: tool_choice
                ``"required"`` must be accompanied by a non-empty ``tools``
                list — vLLM's Hermes parser needs the catalogue to choose
                from). ``None`` (the default) means auto when tools are
                passed and unset otherwise.

        Returns:
            Parsed response with content, reasoning, and tool_calls.

        Raises:
            LLMError: On any communication or parsing failure.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        # Per-call temperature takes precedence over constructor default.
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self._temperature is not None:
            kwargs["temperature"] = self._temperature

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if tools:
            kwargs["tools"] = tools
            # An explicit caller choice (e.g. "required") wins; the
            # default is "auto".
            kwargs["tool_choice"] = tool_choice if tool_choice else "auto"
        elif tool_choice:
            # OpenAI/vLLM reject ``tool_choice="required"`` without a
            # ``tools`` field, so the choice is logged and dropped rather
            # than sent in a payload that returns 400.
            log.warning(
                "llm_client.chat: tool_choice=%r ignored because tools is empty "
                "(vLLM requires a tool catalogue when tool_choice != 'none')",
                tool_choice,
            )

        if user:
            kwargs["user"] = user

        # extra_body carries the vLLM-specific features: the Qwen3
        # ``enable_thinking`` toggle and the optional regex constraint,
        # merged so both can go on the same request.
        #
        # vLLM 0.17 reads the pattern from ``structured_outputs.regex``.
        # A top-level ``guided_regex`` field only produces a warning
        # there and the output is decoded without any constraint, so the
        # kwarg is translated to the shape the server reads.
        extra_body: dict[str, Any] = {}
        if not enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if guided_regex is not None:
            structured = extra_body.setdefault("structured_outputs", {})
            structured["regex"] = guided_regex
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Extract response body from HTTPStatusError for debugging
            detail = ""
            if hasattr(exc, "response"):
                try:
                    detail = f" | Response: {exc.response.text}"  # type: ignore[union-attr]
                except Exception:
                    pass
            has_tools = "tools" in kwargs
            log.error(
                "LLM call failed (model=%s, has_tools=%s): %s%s",
                self._model,
                has_tools,
                exc,
                detail,
            )
            raise LLMError(f"LLM call failed: {exc}{detail}") from exc

        message = response.choices[0].message

        # vLLM's --reasoning-parser qwen3 puts the reasoning tokens on
        # the message object as `reasoning` (vLLM >=0.17) or
        # `reasoning_content` (vLLM <0.17).
        reasoning: str | None = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    # Malformed JSON is wrapped in an error dict so the
                    # loop can append a tool-error result instead of
                    # crashing.
                    arguments = {"_parse_error": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        content = message.content or ""

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            raw={
                "id": response.id,
                "model": response.model,
                "usage": (
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None
                ),
            },
        )

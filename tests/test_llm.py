"""Tests for LLM clients (OpenAILLMClient mock + DummyLLMClient)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from r2c2_agent.agent.interfaces.types import LLMResponse
from r2c2_agent.agent.llm.client import OpenAILLMClient
from r2c2_agent.agent.llm.dummy import RESPONSE_SEQUENCES, DummyLLMClient

# OpenAILLMClient against a mocked OpenAI SDK


def _make_mock_response(
    content: str = "Hello",
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    response_id: str = "resp_1",
    model: str = "test-model",
) -> MagicMock:
    """Build a mock openai ChatCompletion response.

    OpenAILLMClient reads ``message.reasoning`` (the vLLM ≥0.17 field
    name) and falls back to ``message.reasoning_content`` (vLLM <0.17).
    A MagicMock synthesizes any attribute, so an unset
    ``message.reasoning`` would be a MagicMock instead of None and
    assertions on ``result.reasoning`` would fail. Both fields are
    therefore set to the value passed as ``reasoning_content``.
    """
    message = MagicMock()
    message.content = content
    message.reasoning = reasoning_content
    message.reasoning_content = reasoning_content

    if tool_calls:
        mock_tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = json.dumps(tc["arguments"])
            mock_tcs.append(mock_tc)
        message.tool_calls = mock_tcs
    else:
        message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.id = response_id
    response.model = model
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150

    return response


class TestOpenAILLMClient:
    @pytest.mark.asyncio
    async def test_text_only_response(self) -> None:
        mock_response = _make_mock_response(content="The answer is 42")

        with patch("r2c2_agent.agent.llm.client.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_cls.return_value = mock_instance

            client = OpenAILLMClient(base_url="http://test:8000/v1", model="test")
            result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "The answer is 42"
        assert result.reasoning is None
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_with_reasoning(self) -> None:
        mock_response = _make_mock_response(
            content="Spielberg",
            reasoning_content="Let me think about this...",
        )

        with patch("r2c2_agent.agent.llm.client.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_cls.return_value = mock_instance

            client = OpenAILLMClient(base_url="http://test:8000/v1", model="test")
            result = await client.chat(messages=[{"role": "user", "content": "q"}])

        assert result.reasoning == "Let me think about this..."

    @pytest.mark.asyncio
    async def test_with_tool_calls(self) -> None:
        mock_response = _make_mock_response(
            content="",
            tool_calls=[
                {"id": "c1", "name": "search_passages", "arguments": {"query": "Jaws"}},
                {"id": "c2", "name": "get_passage", "arguments": {"passage_id": "p1"}},
            ],
        )

        with patch("r2c2_agent.agent.llm.client.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_cls.return_value = mock_instance

            client = OpenAILLMClient(base_url="http://test:8000/v1", model="test")
            result = await client.chat(
                messages=[{"role": "user", "content": "q"}],
                tools=[{"type": "function", "function": {"name": "search_passages"}}],
            )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "search_passages"
        assert result.tool_calls[0].arguments == {"query": "Jaws"}
        assert result.tool_calls[1].name == "get_passage"

    @pytest.mark.asyncio
    async def test_malformed_tool_call_json(self) -> None:
        """Malformed JSON in tool call arguments → _parse_error key."""
        mock_response = _make_mock_response(content="")
        mock_tc = MagicMock()
        mock_tc.id = "c1"
        mock_tc.function.name = "search_passages"
        mock_tc.function.arguments = "{invalid json"
        mock_response.choices[0].message.tool_calls = [mock_tc]

        with patch("r2c2_agent.agent.llm.client.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_cls.return_value = mock_instance

            client = OpenAILLMClient(base_url="http://test:8000/v1", model="test")
            result = await client.chat(messages=[{"role": "user", "content": "q"}])

        assert len(result.tool_calls) == 1
        assert "_parse_error" in result.tool_calls[0].arguments

    @pytest.mark.asyncio
    async def test_llm_error_raised(self) -> None:
        """Communication failure raises LLMError."""
        from r2c2_agent.agent.exceptions import LLMError

        with patch("r2c2_agent.agent.llm.client.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                side_effect=ConnectionError("refused")
            )
            mock_cls.return_value = mock_instance

            client = OpenAILLMClient(base_url="http://test:8000/v1", model="test")
            with pytest.raises(LLMError, match="refused"):
                await client.chat(messages=[{"role": "user", "content": "q"}])

    @pytest.mark.asyncio
    async def test_raw_contains_usage(self) -> None:
        mock_response = _make_mock_response(content="test")

        with patch("r2c2_agent.agent.llm.client.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_cls.return_value = mock_instance

            client = OpenAILLMClient(base_url="http://test:8000/v1", model="test")
            result = await client.chat(messages=[{"role": "user", "content": "q"}])

        assert result.raw is not None
        assert result.raw["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_guided_regex_uses_vllm_017_structured_outputs_api(self) -> None:
        """Regression test for guided_regex being silently ignored by vLLM 0.17.

        Pre-fix: ``OpenAILLMClient.chat(guided_regex=...)`` put the
        pattern at ``extra_body["guided_regex"]``. vLLM 0.17.1 silently
        ignores that top-level field (emits
        ``The following fields were present in the request but ignored:
        {'guided_regex'}`` WARNING) and decodes unconstrained — the
        Tier-2+ ``Answer: <x> | UNKNOWN`` schema is silently dropped.

        Post-fix: the kwarg is translated to
        ``extra_body["structured_outputs"] = {"regex": ...}`` so the
        constraint is actually enforced by vLLM 0.17+ servers.
        """
        mock_response = _make_mock_response(content="Answer: Paris")

        with patch("r2c2_agent.agent.llm.client.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_cls.return_value = mock_instance

            client = OpenAILLMClient(base_url="http://test:8000/v1", model="test")
            await client.chat(
                messages=[{"role": "user", "content": "q"}],
                enable_thinking=False,
                guided_regex=r"^(Answer: [^\n;]{1,200}|UNKNOWN)$",
            )

            call_kwargs = mock_instance.chat.completions.create.call_args.kwargs

        extra_body = call_kwargs.get("extra_body", {})
        # vLLM 0.17 silently ignores a top-level guided_regex field.
        assert "guided_regex" not in extra_body, (
            "Regression: top-level extra_body['guided_regex'] is silently "
            "ignored by vLLM 0.17+. Use structured_outputs.regex instead."
        )
        # vLLM 0.17 reads the pattern from structured_outputs.regex.
        assert "structured_outputs" in extra_body, (
            "Expected structured_outputs key in extra_body."
        )
        structured = extra_body["structured_outputs"]
        assert isinstance(structured, dict)
        assert structured.get("regex") == r"^(Answer: [^\n;]{1,200}|UNKNOWN)$"
        # enable_thinking=False must coexist with the regex on the same request.
        assert extra_body.get("chat_template_kwargs") == {"enable_thinking": False}


class TestDummyLLMClient:
    @pytest.mark.asyncio
    async def test_simple_two_turn(self) -> None:
        """Simple question: tool call then text answer."""
        client = DummyLLMClient(default_type="simple")
        tools = [{"type": "function", "function": {"name": "search_passages"}}]

        r1 = await client.chat(messages=[], tools=tools)
        assert len(r1.tool_calls) == 1
        assert r1.tool_calls[0].name == "search_passages"

        r2 = await client.chat(messages=[], tools=tools)
        assert r2.tool_calls == []
        assert "Spielberg" in r2.content

    @pytest.mark.asyncio
    async def test_multi_hop_three_turn(self) -> None:
        """Multi-hop: two tool calls then text answer."""
        client = DummyLLMClient(default_type="multi_hop")
        tools = [{"type": "function", "function": {"name": "search_passages"}}]

        r1 = await client.chat(messages=[], tools=tools)
        assert len(r1.tool_calls) == 1

        r2 = await client.chat(messages=[], tools=tools)
        assert len(r2.tool_calls) == 1

        r3 = await client.chat(messages=[], tools=tools)
        assert r3.tool_calls == []
        assert "Mackie" in r3.content

    @pytest.mark.asyncio
    async def test_set_question_type(self) -> None:
        client = DummyLLMClient()
        client.set_question_type("aggregation")
        tools = [{"type": "function", "function": {"name": "search_passages"}}]

        r1 = await client.chat(messages=[], tools=tools)
        assert len(r1.tool_calls) == 1

        r2 = await client.chat(messages=[], tools=tools)
        assert "Streep" in r2.content

    @pytest.mark.asyncio
    async def test_verifier_mode(self) -> None:
        client = DummyLLMClient()
        client.set_verifier_mode(True)

        r = await client.chat(messages=[])
        assert "VERDICT: AGREE" in r.content
        assert "CONFIDENCE_ADJUSTMENT:" in r.content

    @pytest.mark.asyncio
    async def test_no_tools_forces_text(self) -> None:
        """When tools=None (forced synthesis), returns text response."""
        client = DummyLLMClient(default_type="simple")
        r = await client.chat(messages=[], tools=None)
        assert r.tool_calls == []
        assert "Answer:" in r.content

    @pytest.mark.asyncio
    async def test_exhausted_script_repeats_last(self) -> None:
        """Calling beyond script length returns last response."""
        client = DummyLLMClient(default_type="simple")
        tools = [{"type": "function", "function": {"name": "search_passages"}}]

        # Simple has 2 responses
        await client.chat(messages=[], tools=tools)
        await client.chat(messages=[], tools=tools)
        r3 = await client.chat(messages=[], tools=tools)  # Beyond script
        assert r3.tool_calls == []  # Gets last (text) response

    @pytest.mark.asyncio
    async def test_all_question_types_have_sequences(self) -> None:
        """All 6 types + default have response sequences."""
        expected = {
            "simple",
            "simple_w_condition",
            "comparison",
            "aggregation",
            "multi_hop",
            "post_processing",
            "default",
        }
        assert set(RESPONSE_SEQUENCES.keys()) == expected

    @pytest.mark.asyncio
    async def test_nuggets_in_reasoning(self) -> None:
        """Dummy responses include NUGGET tags in reasoning."""
        client = DummyLLMClient(default_type="simple")
        tools = [{"type": "function", "function": {"name": "search_passages"}}]

        await client.chat(messages=[], tools=tools)  # tool call
        r2 = await client.chat(messages=[], tools=tools)  # text
        assert r2.reasoning is not None
        assert "NUGGET[" in r2.reasoning

    @pytest.mark.asyncio
    async def test_confidence_in_reasoning(self) -> None:
        """Dummy responses include Confidence: N in reasoning."""
        client = DummyLLMClient(default_type="simple")
        tools = [{"type": "function", "function": {"name": "search_passages"}}]

        await client.chat(messages=[], tools=tools)
        r2 = await client.chat(messages=[], tools=tools)
        assert r2.reasoning is not None
        assert "Confidence:" in r2.reasoning

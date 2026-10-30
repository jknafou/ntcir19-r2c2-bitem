"""Regression tests for prompt templates.

Covers the cascade-reasoning / alias-first directive.
"""

from __future__ import annotations

from r2c2_agent.agent.prompt import (
    SYNTHESIS_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    build_synthesis_system_prompt,
    build_system_prompt,
)


class TestCascadeNuggetsBlockPhase1:
    def test_block_present_between_polarity_and_quality_bar(self) -> None:
        """Insert position matters — alias directive must come after
        the polarity definition and before the quality bar so the model
        reads them in the intended order."""
        idx_polarity_close = SYSTEM_PROMPT_TEMPLATE.index("</evidence_polarity>")
        idx_cascade_open = SYSTEM_PROMPT_TEMPLATE.index("<cascade_nuggets>")
        idx_cascade_close = SYSTEM_PROMPT_TEMPLATE.index("</cascade_nuggets>")
        idx_quality = SYSTEM_PROMPT_TEMPLATE.index("<nugget_quality_bar>")
        assert idx_polarity_close < idx_cascade_open
        assert idx_cascade_open < idx_cascade_close
        assert idx_cascade_close < idx_quality

    def test_block_teaches_alias_first(self) -> None:
        """The block's value comes from a worked example — strip it and
        the directive collapses to generic platitude."""
        start = SYSTEM_PROMPT_TEMPLATE.index("<cascade_nuggets>")
        end = SYSTEM_PROMPT_TEMPLATE.index("</cascade_nuggets>")
        block = SYSTEM_PROMPT_TEMPLATE[start:end]
        assert "ALIAS nugget FIRST" in block
        assert "substantive nugget" in block
        assert "Mjolnir" in block
        assert "Thor's hammer" in block
        assert "BOTH surface forms" in block

    def test_built_prompt_preserves_block(self) -> None:
        """build_system_prompt interpolates tools — block must survive."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_passages",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        built = build_system_prompt(tools)
        assert "<cascade_nuggets>" in built
        assert "ALIAS nugget FIRST" in built
        # Double-brace escape renders correctly — the worked example
        # should show literal single braces, not the raw `{{...}}` escape.
        assert "{claim:" in built  # literal example rendered
        assert "{{claim:" not in built  # escape itself should not leak


class TestCascadeNuggetsBlockPhase2:
    def test_synthesis_template_carries_alias_directive(self) -> None:
        """Phase 2 also records evidence via record_evidence — the
        alias-first rule must apply there too."""
        assert "<cascade_nuggets>" in SYNTHESIS_PROMPT_TEMPLATE
        assert "</cascade_nuggets>" in SYNTHESIS_PROMPT_TEMPLATE
        assert "ALIAS nugget FIRST" in SYNTHESIS_PROMPT_TEMPLATE

    def test_built_synthesis_prompt_preserves_block(self) -> None:
        tools: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "record_evidence",
                    "description": "rec",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        built = build_synthesis_system_prompt(tools, session_index=1, max_sessions=3)
        assert "<cascade_nuggets>" in built
        assert "ALIAS nugget FIRST" in built

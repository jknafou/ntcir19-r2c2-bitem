"""End-to-end integration test with a real DeBERTa NLI sidecar.

This test is opt-in. It runs only when ``RUN_DEBERTA_INTEGRATION=1``
is set in the environment and the sidecar responds on
``$DEBERTA_NLI_URL``.

The pure-Python unit tests mock ``_deberta_label_probs``, so they
cannot catch HTTP contract bugs between the pipeline and the sidecar
(wrong endpoint path, wrong JSON schema such as a ``query`` field sent
where ``queries`` is expected, timeout misconfiguration). A single
end-to-end test against the real subprocess is the cheapest early
warning for that class of mismatch.

Typical usage:

    # Launch sidecar in another terminal
    python3 deberta_server.py --port 18500 &

    # Run the integration test
    RUN_DEBERTA_INTEGRATION=1 DEBERTA_NLI_URL=http://localhost:18500 \\
        uv run pytest tests/integration/test_ladder_e2e_deberta.py -v

The test is automatically skipped when the env var is not set or
when the sidecar is unreachable, so CI without the sidecar passes
(skipped) while local / HPC runs with the sidecar exercise the full
contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ORCHESTRATOR = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "orchestrator"
)
if str(_ORCHESTRATOR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR))


# Skip conditions


def _env_opt_in() -> bool:
    return os.environ.get("RUN_DEBERTA_INTEGRATION", "").strip() == "1"


def _sidecar_reachable() -> bool:
    """Probe the DeBERTa server with a trivial request. An unreachable
    server skips the test instead of failing it, because CI without the
    sidecar must pass."""
    import json as _json
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    base = os.environ.get("DEBERTA_NLI_URL", "").rstrip("/")
    if not base:
        return False
    payload = _json.dumps(
        {"passage_text": "The sky is blue.", "claim_text": "The sky is blue."}
    ).encode("utf-8")
    req = _urlreq.Request(
        f"{base}/label_probs",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=5.0) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
    except (_urlerr.URLError, TimeoutError, OSError, ValueError):
        return False
    return isinstance(body, dict) and "entailment" in body


# A module-level skipif lets pytest print one concise reason at
# collection time rather than per-test.
_pytestmark = pytest.mark.skipif(
    not _env_opt_in(),
    reason="RUN_DEBERTA_INTEGRATION=1 required to run DeBERTa integration tests",
)


pytestmark = [_pytestmark]


# End-to-end pins


@pytest.fixture(autouse=True)
def _reachable_sidecar() -> None:
    if not _sidecar_reachable():
        pytest.skip("DeBERTa sidecar not reachable at $DEBERTA_NLI_URL")


class TestDebertaHttpContract:
    """Exercises the HTTP contract between ``_deberta_label_probs`` and
    the DeBERTa sidecar. Catches field-name / endpoint-path / schema
    drift between the pipeline call site and the server."""

    def test_basic_entailment_returns_structured_probs(self) -> None:
        from r2c2_agent.orchestrator.pipeline_handlers import _deberta_label_probs

        probs = _deberta_label_probs(
            "Luke Skywalker piloted Red Five at the Battle of Yavin.",
            "Luke piloted Red Five.",
        )
        assert probs is not None, "sidecar returned no probs — HTTP contract regression"
        assert set(probs.keys()) == {"entailment", "neutral", "contradiction"}
        for k, v in probs.items():
            assert 0.0 <= v <= 1.0, f"{k} probability out of bounds: {v}"
        # The two claims are semantically equivalent, so entailment must dominate.
        assert probs["entailment"] > probs["contradiction"], probs

    def test_clear_contradiction_detected(self) -> None:
        from r2c2_agent.orchestrator.pipeline_handlers import _deberta_label_probs

        probs = _deberta_label_probs(
            "Luke Skywalker piloted Red Five.",
            "Han Solo piloted Red Five.",
        )
        assert probs is not None
        # Clear polarity swap — contradiction should outweigh entailment.
        assert probs["contradiction"] > probs["entailment"], probs


class TestAnswerSearchSupportE2E:
    """End-to-end smoke through ``answer_search_support``'s primitive
    stack (retrieval is mocked; DeBERTa is real). Validates that an
    auto-commit Nugget lands on the DB when the real sidecar entails
    the claim."""

    def test_auto_commit_fires_on_real_entailment(self) -> None:
        import asyncio

        from r2c2_agent.orchestrator.pipeline_handlers import answer_search_support, seed_db_empty

        class _MCP:
            async def call_tool(self, name: str, args: dict) -> dict:
                return {
                    "passages": [
                        {
                            "passage_id": "P1",
                            "text": (
                                "Luke Skywalker piloted Red Five at the "
                                "Battle of Yavin, a decisive moment in the "
                                "Galactic Civil War."
                            ),
                            "score": 0.95,
                        }
                    ]
                }

        db = seed_db_empty("qid_e2e", "Who piloted Red Five?", "simple")
        mcp = _MCP()
        verified, details = asyncio.run(
            answer_search_support(db, "Luke Skywalker", mcp, origin="integration")
        )
        assert verified is True
        # At least one span should have cleared the auto-commit band
        # (p_entail >= 0.85, p_contra < 0.25). If the real sidecar
        # disagrees there are no auto-committed nuggets, and the
        # details go into the failure message.
        assert details.get("auto_nuggets"), (
            f"expected at least one auto-committed nugget; got details={details}"
        )

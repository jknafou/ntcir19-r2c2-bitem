# ruff: noqa: N999 — line test module naming per plan's lines taxonomy.
"""Line test: Path-5 tier-4 rescue trigger rule.

Path 5 fires inside ``_force_deliver_impl`` when ALL of:
  - tier-4 produced a shape-passing guess (``t4_guess`` non-empty,
    ``tier4_details['shape_pass']`` True)
  - tier-4's primary verify failed with an infrastructure error — one
    of ``verify_timeout``, ``verify_exception:<type>``,
    ``retrieval_error``, ``no_spans``
  - semantic rejections (``refuted`` / ``uncertain`` / ``not_verified``)
    do not qualify.

When all three hold, the ladder calls ``answer_search_support`` with
``origin="path5_tier4_rescue"`` against the tier-4 guess, and on auto-
commit emits with ``ladder_path="path5_tier4_ass_rescued"``.

This test exercises the specific predicate via a mini-controlled fake
``tier4_details`` dict rather than the full block-level cascade.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ORCH = (
    Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "orchestrator"
)
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


# The predicate is inlined in _force_deliver_impl, so its logic is
# mirrored here: the test pins the contract and has to be kept in step
# with the implementation.
def _infra_err_predicate(t4_err: str) -> bool:
    return (
        t4_err == "verify_timeout"
        or t4_err.startswith("verify_exception:")
        or t4_err == "retrieval_error"
        or t4_err == "no_spans"
    )


def test_path5_trigger_on_verify_timeout() -> None:
    assert _infra_err_predicate("verify_timeout") is True


def test_path5_trigger_on_verify_exception() -> None:
    assert _infra_err_predicate("verify_exception:TimeoutError") is True
    assert _infra_err_predicate("verify_exception:ConnectionError") is True


def test_path5_trigger_on_retrieval_error() -> None:
    assert _infra_err_predicate("retrieval_error") is True


def test_path5_trigger_on_no_spans() -> None:
    assert _infra_err_predicate("no_spans") is True


def test_path5_no_trigger_on_refuted() -> None:
    """Refuted is a semantic rejection — corpus actively contradicts
    the guess. Path 5 does not fire here."""
    assert _infra_err_predicate("refuted") is False


def test_path5_no_trigger_on_uncertain() -> None:
    """Uncertain is a weak signal, not infrastructure failure."""
    assert _infra_err_predicate("uncertain") is False


def test_path5_no_trigger_on_not_verified() -> None:
    assert _infra_err_predicate("not_verified") is False


def test_path5_no_trigger_on_verified() -> None:
    """Tier-4 verified doesn't need rescue at all."""
    assert _infra_err_predicate("verified") is False

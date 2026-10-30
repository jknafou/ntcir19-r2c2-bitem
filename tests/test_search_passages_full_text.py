"""Pin the text caps of ``search_passages``.

Untruncated passages overflow the vLLM 131K context once many searches
accumulate, so ``search_passages`` truncates at
``SEARCH_MAX_TEXT_LENGTH`` (2000 chars). That is much larger than the
800-char cap used by ``get_passage`` / ``extract_from_passage``, and
bounded enough that the preflight budget gate in the driver never has
to fire.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SERVER_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "r2c2_agent" / "mcp_server" / "server.py"
)


def _load_server_module():
    """Load src/r2c2_agent/mcp_server/server.py for unit tests.

    Avoids importing the FastAPI app (which needs FAISS + ES); we only
    need ``_build_passage_entry`` and ``MAX_TEXT_LENGTH`` for this test.
    """
    spec = importlib.util.spec_from_file_location("mcp_r2c2_server", _SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcp_r2c2_server"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def server_mod():
    try:
        return _load_server_module()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"mcp-r2c2 server module not importable in this env: {e}")


def _long_text() -> str:
    # >> MAX_TEXT_LENGTH (800) sentences; ends well past any cap.
    sentence = (
        "The zeroth law of robotics, added by Asimov later than the three, "
        "states that a robot may not harm humanity. "
    )
    return sentence * 25  # ≈ 2400 chars


class TestPassageEntryTruncation:
    def test_truncate_false_preserves_full_text(self, server_mod) -> None:
        src = {"contents": _long_text(), "title": "T", "section": "S", "source": "wiki"}
        entry = server_mod._build_passage_entry(
            "Q123__00001",
            src,
            rank=1,
            score=1.0,
            retrieval_mode="hybrid",
            truncate=False,
        )
        assert entry["text"] == _long_text()
        assert len(entry["text"]) > server_mod.MAX_TEXT_LENGTH
        assert not entry["text"].endswith("...")

    def test_truncate_true_still_caps_text(self, server_mod) -> None:
        """Call sites that opt in to truncation (e.g. the
        get_passage/extract_from_passage helpers) get the capped
        behaviour. ``_soft_truncate`` extends past ``max_len``
        only as far as the next sentence boundary, with a hard cap at
        ``2 * max_len`` (see server._soft_truncate)."""
        src = {"contents": _long_text(), "title": "T", "section": "S", "source": "wiki"}
        entry = server_mod._build_passage_entry(
            "Q123__00001",
            src,
            rank=1,
            score=1.0,
            retrieval_mode="hybrid",
            truncate=True,
        )
        assert len(entry["text"]) < len(_long_text())  # truncation actually happened
        assert len(entry["text"]) <= 2 * server_mod.MAX_TEXT_LENGTH


class TestSearchPassagesCallSite:
    """Guard the call-site contract: ``search_passages`` passes the
    larger ``SEARCH_MAX_TEXT_LENGTH`` cap (not the default 800-char one)
    so the model sees enough per-hit context while staying bounded.
    """

    def test_search_passages_uses_search_max_text_length(self) -> None:
        src = _SERVER_PATH.read_text(encoding="utf-8")
        marker = "async def search_passages("
        idx = src.find(marker)
        assert idx >= 0, "search_passages definition missing"
        next_def = src.find("\nasync def ", idx + len(marker))
        if next_def == -1:
            next_def = src.find("\ndef ", idx + len(marker))
        if next_def == -1:
            next_def = len(src)
        body = src[idx:next_def]
        assert "_build_passage_entry(" in body, "call to _build_passage_entry missing"
        # The cap is passed via the ``max_len`` kwarg and is much larger
        # than the default 800, so the context stays within budget.
        assert "max_len=SEARCH_MAX_TEXT_LENGTH" in body, (
            "search_passages must pass max_len=SEARCH_MAX_TEXT_LENGTH "
            "so per-hit text is bounded (F++ hardening) — regression detected"
        )

    def test_constants_exist(self) -> None:
        """Pin the two text-cap constants so future edits cannot silently
        remove them."""
        src = _SERVER_PATH.read_text(encoding="utf-8")
        assert "MAX_TEXT_LENGTH = 800" in src
        assert "SEARCH_MAX_TEXT_LENGTH = 2000" in src
        assert "DEFAULT_MAX_RESULTS = 10" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

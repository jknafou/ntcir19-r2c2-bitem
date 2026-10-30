"""Thin async client for the CP2 verifier (Qwen3.5-27B-FP8).

The verifier is a second vLLM process, launched in the same job as the
main 122B model by the worker's Slurm script, that grades each draft
answer against the retrieved passages. Its verdict decides whether to
inject a reflection turn back into the agentic loop or to let the draft
stand.

Fail-open contract:
    :meth:`VerifierClient.verify_draft` never raises. On any
    HTTP / parsing / timeout error it returns
    ``VerifierVerdict(verdict="ok", confidence=0.0, reason="error: ...")``.
    The downstream trigger rule requires ``verdict != "ok"`` and
    ``confidence >= 0.7`` to fire, so a fail-open verdict is a no-op
    and the loop runs without reflection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

__all__ = ["VerifierClient", "VerifierVerdict"]

log = logging.getLogger(__name__)


_VERDICT_RE = re.compile(r"^\s*verdict\s*:\s*(\w+)", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(
    r"^\s*confidence\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE | re.MULTILINE
)
_REASON_RE = re.compile(r"^\s*reason\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

_ALLOWED_VERDICTS: frozenset[str] = frozenset({"ok", "mismatch", "unsupported"})


@dataclass
class VerifierVerdict:
    """Structured output from a single verifier call.

    Attributes:
        verdict: One of ``"ok"`` / ``"mismatch"`` / ``"unsupported"``.
            ``"ok"`` means the draft answer matches the evidence and
            gold; ``"mismatch"`` means the draft contradicts the
            retrieved passages or the gold; ``"unsupported"`` means
            the draft is not supported by the retrieved passages but
            isn't clearly wrong either (often: agent answered from
            parametric knowledge without evidence).
        confidence: ``0.0 – 1.0`` float, the verifier's self-reported
            confidence in its verdict. The CP2 trigger threshold is
            ``0.7`` — ``mismatch``/``unsupported`` below ``0.7`` does
            NOT fire a reflection injection.
        reason: Free-text one-liner the verifier produces, used for
            debug logging and the audit trail.
    """

    verdict: str
    confidence: float
    reason: str = ""


class VerifierClient:
    """Async client around a vLLM-served ``Qwen3.5-27B-FP8`` verifier.

    Single public method: :meth:`verify_draft`. No caching (each call
    is unique given the draft). No retry: on any error the method
    returns an ``ok`` verdict at confidence ``0.0`` so the caller
    defaults to not injecting a reflection (fail-open).

    Args:
        base_url: vLLM endpoint (e.g. ``"http://127.0.0.1:18004/v1"``).
        model: Model identifier. Must match what vLLM was launched with.
        timeout: Per-request timeout in seconds. Defaults to 30s.

    Raises:
        ValueError: If ``base_url`` is empty.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "Qwen/Qwen3.5-27B-FP8",
        timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("VerifierClient: base_url must be non-empty")
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy",
            timeout=timeout,
        )
        self._model = model

    async def verify_draft(
        self,
        question: str,
        expected: str,
        draft: str,
        passages: list[str],
        *,
        max_passage_chars: int = 600,
    ) -> VerifierVerdict:
        """Grade ``draft`` against ``expected`` and retrieved passages.

        Args:
            question: The original user question text.
            expected: The gold reference answer.
            draft: The agent's draft answer.
            passages: Recent tool-result strings (already windowed by
                the caller). Only the last 3 are consulted.
            max_passage_chars: Per-passage truncation to keep the
                verifier prompt small.

        Returns:
            A :class:`VerifierVerdict`. Never raises — on error,
            returns ``("ok", 0.0, "error: ...")``.
        """
        trimmed = [p[:max_passage_chars] for p in passages[-3:]]
        prompt = self._build_prompt(question, expected, draft, trimmed)
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            log.warning("VerifierClient.verify_draft: %s — failing open", e)
            return VerifierVerdict(verdict="ok", confidence=0.0, reason=f"error: {e}")

        text = (resp.choices[0].message.content or "").strip()
        return self._parse(text)

    @staticmethod
    def _build_prompt(
        question: str,
        expected: str,
        draft: str,
        passages: list[str],
    ) -> str:
        """Build the verifier prompt body.

        The format intentionally asks for three fixed lines so the
        regex parser can extract verdict / confidence / reason
        without needing a JSON schema.

        Args:
            question: Original user question.
            expected: Gold reference answer.
            draft: Agent draft answer.
            passages: Already-trimmed recent tool-result strings.

        Returns:
            A single-string prompt ready to send as a user message.
        """
        if passages:
            passages_joined = "\n\n".join(
                f"[{i + 1}] {p}" for i, p in enumerate(passages)
            )
        else:
            passages_joined = "(no passages retrieved)"
        return (
            "You are a fact-checker grading a draft answer against "
            "retrieved evidence and a gold reference answer.\n\n"
            "Grade the draft answer on three axes:\n"
            "  1. Does the draft answer the question that was asked?\n"
            "  2. Is the draft consistent with the retrieved passages?\n"
            "  3. Does the draft agree with the gold reference answer?\n\n"
            "Verdict rules:\n"
            '  - "ok": draft matches gold AND is grounded in the '
            "retrieved passages\n"
            '  - "mismatch": draft contradicts gold OR contradicts '
            "the retrieved passages\n"
            '  - "unsupported": draft does not contradict anything '
            "but cannot be verified from the retrieved passages "
            "(likely answered from general knowledge, needs more "
            "retrieval)\n\n"
            "Reply in EXACTLY this format (three lines, no other text):\n"
            "verdict: <ok|mismatch|unsupported>\n"
            "confidence: <float 0.0-1.0>\n"
            "reason: <one short sentence>\n\n"
            "---\n"
            f"Question: {question}\n\n"
            f"Gold reference answer: {expected}\n\n"
            f"Draft answer: {draft}\n\n"
            "Retrieved passages:\n"
            f"{passages_joined}\n"
            "---\n"
        )

    @staticmethod
    def _parse(text: str) -> VerifierVerdict:
        """Parse a verifier reply into a :class:`VerifierVerdict`.

        Expected format:
            verdict: <ok|mismatch|unsupported>
            confidence: <float>
            reason: <text>

        On any parse failure, returns ``("ok", 0.0, "parse_failed: ...")``
        so the fail-open contract is preserved.

        Args:
            text: Raw verifier reply (already stripped of whitespace).

        Returns:
            Parsed verdict. Never raises.
        """
        if not text:
            return VerifierVerdict(
                verdict="ok", confidence=0.0, reason="parse_failed: empty"
            )

        verdict_m = _VERDICT_RE.search(text)
        if verdict_m is None:
            return VerifierVerdict(
                verdict="ok",
                confidence=0.0,
                reason=f"parse_failed: no verdict in {text[:80]!r}",
            )
        verdict_raw = verdict_m.group(1).lower().strip()
        if verdict_raw not in _ALLOWED_VERDICTS:
            return VerifierVerdict(
                verdict="ok",
                confidence=0.0,
                reason=f"parse_failed: bad verdict {verdict_raw!r}",
            )

        confidence = 0.0
        conf_m = _CONFIDENCE_RE.search(text)
        if conf_m is not None:
            try:
                confidence = float(conf_m.group(1))
            except ValueError:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

        reason = ""
        reason_m = _REASON_RE.search(text)
        if reason_m is not None:
            reason = reason_m.group(1).strip()

        return VerifierVerdict(
            verdict=verdict_raw, confidence=confidence, reason=reason
        )

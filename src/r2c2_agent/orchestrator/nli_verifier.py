"""Grounded-factuality NLI oracle server — v2 post-trace pipeline.

Serves a localhost HTTP endpoint `POST /check_entailment` that given a passage
and a short claim, returns {label, score} where:
  - `label ∈ {"entails", "bogus"}` (binary; "bogus" collapses NLI "neutral" +
    "contradicts" per MiniCheck's training semantics).
  - `score ∈ [0, 1]` = P(claim is supported by passage).

Threshold is 0.5 by default (MiniCheck paper calibration, Tang et al. EMNLP
2024 arXiv:2404.10774).

Pluggable backend via `NLI_BACKEND` env var:
  - `minicheck` (default) → `lytang/MiniCheck-Flan-T5-Large`
  - `modernce`            → `dleemiller/ModernCE-large-nli` (sentence-transformers)

Both are CPU-friendly. Used by the co-located worker job; not exposed to the
orchestrator or other nodes. Launched from `slurm/orchestrator/worker.sbatch`
alongside existing MCP/ES processes (Phase 5).

Launch:
    python3 src/r2c2_agent/orchestrator/nli_verifier.py --port 18500

Health:
    GET http://localhost:18500/health → {"status": "ok", "backend": ..., "model": ...}

Inference:
    POST http://localhost:18500/check_entailment
    Body: {"passage_text": "...", "claim_text": "..."}
    → {"label": "entails"|"bogus", "score": 0.87}

Graceful fallback: on inference error returns `{"label": "bogus", "score": 0.0}`
— the safe-conservative default for the S3 bogus filter (§4.6 design doc).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Protocol

from aiohttp import web

log = logging.getLogger(__name__)

# Typed app-state key (avoids NotAppKeyWarning from aiohttp>=3.10).
_BACKEND_KEY: web.AppKey[NLIBackend] = web.AppKey("backend", object)


SUPPORT_THRESHOLD = 0.5  # MiniCheck paper default
DEFAULT_PORT = 18500
DEFAULT_BACKEND = "minicheck"

MINICHECK_MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
MODERNCE_MODEL_ID = "dleemiller/ModernCE-large-nli"
# DeBERTa-v3-large-MNLI is the tiebreak for the uncertain band of the
# hybrid cascade. It scores a paraphrased claim as entailed (p_entail
# >= 0.99) in cases where MiniCheck systematically underscores it.
DEBERTA_MNLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Pre-cached local paths (override via NLI_MODEL_CACHE_DIR; HuggingFace
# Hub is the fallback if the local directory is absent).
_NLI_CACHE_ROOT = os.environ.get("NLI_MODEL_CACHE_DIR", "models")
_MINICHECK_LOCAL_DIR = f"{_NLI_CACHE_ROOT}/minicheck-flan-t5-large"
_MODERNCE_LOCAL_DIR = f"{_NLI_CACHE_ROOT}/modernce-large-nli"
_DEBERTA_MNLI_LOCAL_DIR = f"{_NLI_CACHE_ROOT}/deberta-v3-large-mnli"

# Max input chars (truncate passages/claims to avoid OOM on very long inputs).
_PASSAGE_MAX_CHARS = 8000
_CLAIM_MAX_CHARS = 1000


@dataclass
class ScoreResult:
    label: str  # "entails" or "bogus"
    score: float


# Backend protocol


class NLIBackend(Protocol):
    """All NLI backends expose the same single-call scoring interface.

    Backends that produce a full 3-class softmax (entailment / neutral /
    contradiction) MAY additionally implement ``label_probs`` — callers
    use ``hasattr(backend, "label_probs")`` before issuing that call so
    collapsed-head backends (MiniCheck) continue to work.
    """

    name: str
    model_id: str

    def score(self, passage: str, claim: str) -> float:
        """Return P(claim is supported by passage) ∈ [0, 1]."""
        ...


# MiniCheck backend (Flan-T5-Large via HuggingFace Transformers)


class MiniCheckBackend:
    """MiniCheck-Flan-T5-Large from Tang et al. EMNLP 2024 (arXiv:2404.10774).

    Input format (from the paper §3 and the HF model card):
        "predict: <document> </s> <claim>"
    The model emits a single token "0" or "1"; we softmax the two logits to get
    a calibrated P(supported).
    """

    name = "minicheck"
    model_id = MINICHECK_MODEL_ID

    def __init__(self, model_path: str | None = None) -> None:
        # Lazy heavyweight imports so the module can be imported in tests
        # without transformers/torch being installed.
        import torch  # type: ignore[import]
        from transformers import (  # type: ignore[import]
            T5ForConditionalGeneration,
            T5Tokenizer,
        )

        self._torch = torch
        path = model_path or _MINICHECK_LOCAL_DIR
        if not os.path.isdir(path):
            # Fall back to HF hub ID; transformers will download + cache.
            path = MINICHECK_MODEL_ID
            log.info("minicheck: local dir not found, loading from hub: %s", path)

        log.info("minicheck: loading tokenizer + model from %s ...", path)
        self._tokenizer = T5Tokenizer.from_pretrained(path)
        self._model = T5ForConditionalGeneration.from_pretrained(path)
        self._model.eval()
        # MiniCheck-Flan-T5-Large label token IDs: 3 (no-support) / 209 (support).
        # These come from the checkpoint's fine-tuning labels, not from the
        # digit-vocabulary items "0"/"1" (which tokenize to IDs 632/536). See
        # https://github.com/Liyan06/MiniCheck/blob/main/minicheck/inference.py
        # With the digit IDs the binary softmax reads two unrelated logits near
        # -10 and scores about 0.003 on every input, whatever the entailment.
        self._label_token_ids = [3, 209]
        log.info("minicheck: ready")

    def score(self, passage: str, claim: str) -> float:
        torch = self._torch
        passage = (passage or "")[:_PASSAGE_MAX_CHARS]
        claim = (claim or "")[:_CLAIM_MAX_CHARS]
        prompt = f"predict: {passage} </s> {claim}"
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # First generated-token logits over the vocab
        logits = outputs.scores[0][0]  # shape [vocab_size]
        zero_logit = float(logits[self._label_token_ids[0]])
        one_logit = float(logits[self._label_token_ids[1]])
        # Binary softmax over just the two label tokens
        max_l = max(zero_logit, one_logit)
        e0 = pow(2.718281828, zero_logit - max_l)
        e1 = pow(2.718281828, one_logit - max_l)
        p_support = e1 / (e0 + e1)
        return float(p_support)


# ModernCE backend (sentence-transformers cross-encoder, MNLI-trained)


class ModernCEBackend:
    """ModernCE-large-nli from dleemiller — generic MNLI-style NLI.

    Returns a 3-label softmax [entail, neutral, contradict]. We collapse to a
    single P(supported) = p_entail (per the §2.7 decision table: neutral and
    contradict both map to bogus). This is the "simple" backend used in commit
    52b806f's original `nli_verifier.py` implementation.
    """

    name = "modernce"
    model_id = MODERNCE_MODEL_ID

    def __init__(self, model_path: str | None = None) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import]

        path = model_path or _MODERNCE_LOCAL_DIR
        if not os.path.isdir(path):
            path = MODERNCE_MODEL_ID
            log.info("modernce: local dir not found, loading from hub: %s", path)
        log.info("modernce: loading CrossEncoder from %s ...", path)
        self._model = CrossEncoder(path, device="cpu")
        log.info("modernce: ready")

    def score(self, passage: str, claim: str) -> float:
        passage = (passage or "")[:_PASSAGE_MAX_CHARS]
        claim = (claim or "")[:_CLAIM_MAX_CHARS]
        # sentence-transformers CrossEncoder.predict returns a numpy array.
        # For a 3-class MNLI head, shape is [1, 3]: [entail, neutral, contradict].
        scores = self._model.predict([(passage, claim)], apply_softmax=True)
        if hasattr(scores, "shape") and len(scores.shape) == 2:
            return float(scores[0][0])  # P(entail)
        # Some checkpoints emit a single logit → treat as entailment prob.
        return float(scores[0])


# DeBERTa-v3-MNLI backend (uncertain-band tiebreak for the hybrid cascade)


class DebertaMNLIBackend:
    """DeBERTa-v3-large-MNLI head served as the hybrid-cascade tiebreak.

    Returns p_entail from the 3-class softmax. Label mapping confirmed by
    the spike at ``spikes/deberta_mnli_probe.py`` ::

        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    Tokenization is the standard MNLI pair form — ``tokenizer(premise,
    hypothesis, truncation=True, max_length=512)``. Runs on CPU by
    default (300-800 ms/call under MKL-DNN on a dedicated node). The
    login-node spike ran at ~100 s/call (shared CPU, no vectorization)
    — do NOT extrapolate latency from the spike log.
    """

    name = "deberta_mnli"
    model_id = DEBERTA_MNLI_MODEL_ID

    def __init__(self, model_path: str | None = None) -> None:
        import torch  # type: ignore[import]
        from transformers import (  # type: ignore[import]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        path = model_path or _DEBERTA_MNLI_LOCAL_DIR
        if not os.path.isdir(path):
            path = DEBERTA_MNLI_MODEL_ID
            log.info("deberta_mnli: local dir not found, loading from hub: %s", path)
        log.info("deberta_mnli: loading tokenizer + model from %s ...", path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        # CPU-only by design, like MiniCheck: vLLM takes the GPU, and
        # running this head on CPU avoids competing with the main model
        # for VRAM. Expect a p50 of about 300-800 ms per call under
        # MKL-DNN.
        self._device = "cpu"
        self._model.to(self._device)
        self._model.eval()
        # Resolve the entailment ID from the model config: MNLI
        # checkpoints do not agree on the order of the labels.
        label2id = getattr(self._model.config, "label2id", {}) or {}
        self._entail_id = (
            label2id.get("entailment")
            or label2id.get("ENTAILMENT")
            or label2id.get("entail")
            or 0
        )
        log.info(
            "deberta_mnli: ready (device=%s, entailment_id=%d)",
            self._device,
            int(self._entail_id),
        )

    def _forward(self, passage: str, claim: str):
        torch = self._torch
        passage = (passage or "")[:_PASSAGE_MAX_CHARS]
        claim = (claim or "")[:_CLAIM_MAX_CHARS]
        inputs = self._tokenizer(
            passage,
            claim,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits.squeeze(0)
        return torch.softmax(logits, dim=-1)

    def score(self, passage: str, claim: str) -> float:
        probs = self._forward(passage, claim)
        return float(probs[self._entail_id].item())

    def label_probs(self, passage: str, claim: str) -> dict[str, float]:
        """Return the full 3-class softmax keyed by label string."""
        probs = self._forward(passage, claim)
        id2label = getattr(self._model.config, "id2label", {}) or {}
        out: dict[str, float] = {}
        for i in range(probs.shape[0]):
            lbl = id2label.get(i, str(i))
            if isinstance(lbl, str):
                out[lbl.lower()] = float(probs[i].item())
        return out


# Backend registry


def make_backend(name: str) -> NLIBackend:
    name = (name or "").lower().strip()
    if name == "minicheck":
        return MiniCheckBackend()
    if name == "modernce":
        return ModernCEBackend()
    if name == "deberta_mnli":
        return DebertaMNLIBackend()
    raise ValueError(
        f"unknown NLI backend {name!r}; supported: minicheck, modernce, deberta_mnli"
    )


def score_to_result(score: float) -> ScoreResult:
    """Threshold a [0,1] support probability into the binary label."""
    label = "entails" if score >= SUPPORT_THRESHOLD else "bogus"
    return ScoreResult(label=label, score=float(score))


# aiohttp server


async def _handle_health(request: web.Request) -> web.Response:
    backend: NLIBackend = request.app[_BACKEND_KEY]
    return web.json_response(
        {
            "status": "ok",
            "backend": backend.name,
            "model": backend.model_id,
            "threshold": SUPPORT_THRESHOLD,
        }
    )


async def _handle_check_entailment(request: web.Request) -> web.Response:
    """POST /check_entailment

    Body: {"passage_text": str, "claim_text": str}
    → {"label": "entails"|"bogus", "score": float}
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "invalid_json", "label": "bogus", "score": 0.0},
            status=400,
        )
    passage = body.get("passage_text", "") or ""
    claim = body.get("claim_text", "") or ""
    if not isinstance(passage, str) or not isinstance(claim, str):
        return web.json_response(
            {"error": "types", "label": "bogus", "score": 0.0},
            status=400,
        )
    if not passage.strip() or not claim.strip():
        # Empty inputs get the conservative label: nothing supports the claim.
        return web.json_response({"label": "bogus", "score": 0.0})

    backend: NLIBackend = request.app[_BACKEND_KEY]

    try:
        # Run synchronous model inference off the event loop so concurrent
        # requests don't serialize on a single blocking call.
        score = await asyncio.to_thread(backend.score, passage, claim)
    except Exception as exc:
        log.exception("check_entailment failed: %s", exc)
        # The server never raises to the caller; a failed inference
        # returns the conservative bogus label.
        return web.json_response({"label": "bogus", "score": 0.0})

    # Clamp score to [0, 1] in case a backend returns NaN/out-of-range.
    if not (score == score):  # NaN check without math.isnan
        score = 0.0
    score = max(0.0, min(1.0, float(score)))
    result = score_to_result(score)
    return web.json_response({"label": result.label, "score": result.score})


async def _handle_label_probs(request: web.Request) -> web.Response:
    """POST /label_probs

       Body: {"passage_text": str, "claim_text": str}
       → {"entailment": float, "neutral": float, "contradiction": float}

       Backends that only produce a collapsed binary score (MiniCheck)
       return 501; callers must gate on this. The hybrid cascade
    uses this endpoint to replace the handcrafted polarity /
       entity / quantifier rules with contradiction-probability triggers.
    """
    backend: NLIBackend = request.app[_BACKEND_KEY]
    if not hasattr(backend, "label_probs"):
        return web.json_response(
            {"error": "label_probs_unsupported", "backend": backend.name},
            status=501,
        )
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    passage = body.get("passage_text", "") or ""
    claim = body.get("claim_text", "") or ""
    if (
        not isinstance(passage, str)
        or not isinstance(claim, str)
        or not passage.strip()
        or not claim.strip()
    ):
        # Conservative: treat empty/invalid as pure neutral.
        return web.json_response(
            {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
        )
    try:
        probs = await asyncio.to_thread(backend.label_probs, passage, claim)
    except Exception as exc:
        log.exception("label_probs failed: %s", exc)
        return web.json_response(
            {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
        )
    # Clamp, and ensure all three keys are present. NaN/Inf are scrubbed
    # before clamping: Python's min/max propagate NaN, and json.dumps
    # emits literal "NaN"/"Infinity", which is invalid JSON. aiohttp then
    # sends a 200 with a zero-length body, because the body cannot be
    # encoded.
    import json as _json  # noqa: PLC0415 — inline to survive lint churn
    import math as _math  # noqa: PLC0415

    out = {
        "entailment": 0.0,
        "neutral": 0.0,
        "contradiction": 0.0,
    }
    for k, v in probs.items():
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if not _math.isfinite(fv):
            fv = 0.0
        key = k.lower()
        if key in out:
            out[key] = max(0.0, min(1.0, fv))
    # Pre-encode with allow_nan=False (strict) and return an explicit
    # Response with Content-Length, in case web.json_response defers
    # body serialization past response-finalize (aiohttp on Python 3.14).
    payload = _json.dumps(out, allow_nan=False).encode("utf-8")
    return web.Response(
        body=payload,
        status=200,
        content_type="application/json",
        headers={"Content-Length": str(len(payload))},
    )


def build_app(backend: NLIBackend) -> web.Application:
    app = web.Application()
    app[_BACKEND_KEY] = backend
    app.add_routes(
        [
            web.get("/health", _handle_health),
            web.post("/check_entailment", _handle_check_entailment),
            web.post("/label_probs", _handle_label_probs),
        ]
    )
    return app


# CLI entrypoint


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    description = (__doc__ or "NLI oracle server").splitlines()[0]
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NLI_PORT", DEFAULT_PORT)),
        help="TCP port to bind (default: $NLI_PORT or 18500)",
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("NLI_HOST", "127.0.0.1"),
        help="Bind address (default: 127.0.0.1 — worker-local only)",
    )
    ap.add_argument(
        "--backend",
        default=os.environ.get("NLI_BACKEND", DEFAULT_BACKEND),
        choices=["minicheck", "modernce", "deberta_mnli"],
        help=(
            "NLI backend (default: $NLI_BACKEND or 'minicheck'). "
            "Use 'modernce' as the A/B fallback, 'deberta_mnli' to serve "
            "the hybrid-cascade tiebreak on a separate port."
        ),
    )
    ap.add_argument(
        "--log-level",
        default=os.environ.get("NLI_LOG_LEVEL", "INFO"),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info(
        "nli_verifier: starting backend=%s host=%s port=%d",
        args.backend,
        args.host,
        args.port,
    )
    try:
        backend = make_backend(args.backend)
    except Exception as exc:
        log.exception("nli_verifier: failed to load backend: %s", exc)
        return 2
    app = build_app(backend)
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    except KeyboardInterrupt:  # pragma: no cover
        log.info("nli_verifier: shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())

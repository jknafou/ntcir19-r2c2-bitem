#!/usr/bin/env python3
"""GPU calibration: find the real max concurrency at 100K tokens per request.

Connects to a running vLLM worker and sends progressively more concurrent
100K-token requests until the GPU fails (OOM, preemption, timeout).

Usage (run from the orchestrator node or any node that can reach the worker):

    # Single worker probe:
    python3 src/r2c2_agent/orchestrator/calibrate_gpu.py \
        --url http://localhost:26048/v1 \
        --tokens 100000 \
        --max-concurrency 12 \
        --output slurm/orchestrator/gpu_calibration.json

    # The script will:
    # 1. Send 1 request of 100K tokens → pass/fail
    # 2. If pass: send 2 concurrent → pass/fail
    # 3. Keep incrementing until failure
    # 4. Write the last passing concurrency to the calibration file

The "100K tokens" request is a real chat completion with a padded prompt
that forces the model to allocate ~100K tokens of KV cache. The response
is short (max_tokens=10) so generation cost is negligible — we're purely
testing KV cache capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# Only configure logging when run as a standalone script (not when imported).
# When imported by the orchestrator, print() with [calibrate] prefix is used
# instead so output appears in the orchestrator's stdout log.
_standalone = __name__ == "__main__"
if _standalone:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

# The prompt is approximately `target_tokens` tokens long.
# Measured ratio for the Qwen3.5-122B tokenizer on repeated English text:
# 400K chars → 86K tokens → 4.65 chars/token. The prompt targets slightly
# fewer tokens than requested to stay under max_model_len: the caller
# passes 100K, which is also the vLLM ceiling, and a request that reaches
# it is rejected.
_PAD_SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank. "
_CHARS_PER_TOKEN = 4.42  # measured from shuffled text (lower than 4.65 for ordered)


def build_padded_prompt(target_tokens: int, unique_id: int = 0) -> str:
    """Build a UNIQUE prompt of approximately ``target_tokens * 0.98`` tokens.

    Each call with a different ``unique_id`` produces a prompt with
    different content, defeating vLLM's prefix caching. Without this,
    concurrent identical prompts share KV cache and the calibration
    measures cache efficiency instead of KV capacity.

    The 2% headroom ensures the prompt fits inside ``max_model_len``
    even when ``target_tokens == max_model_len``.
    """
    import random as _rng

    # 5% headroom: the raw prompt tokens get inflated by ~2K tokens of chat
    # template overhead (system prompt wrapper, special tokens, BOS/EOS).
    # At 100K target: 95K raw + ~2K template + 10 output = ~97K total,
    # safely under the 100K max_model_len ceiling.
    safe_target = int(target_tokens * 0.95)
    # Unique header per request — different first tokens defeat prefix cache
    session_id = _rng.Random(unique_id).randint(10000, 99999)
    header = (
        f"You are assistant #{unique_id}. Session {session_id}. "
        f"Answer the following question in one word only. "
        f"The rest of this message is padding to test "
        f"context length capacity.\n\n"
    )
    target_chars = int((safe_target - 50) * _CHARS_PER_TOKEN)
    # Shuffle padding order per unique_id to ensure no shared prefixes
    r = _rng.Random(unique_id)
    words = _PAD_SENTENCE.split()
    sentences = []
    sentence_len = 0
    while sentence_len < target_chars:
        shuffled = words.copy()
        r.shuffle(shuffled)
        s = " ".join(shuffled) + ". "
        sentences.append(s)
        sentence_len += len(s)
    padding = "".join(sentences)[:target_chars]
    footer = (
        f"\n\nQuestion: What is {2 + unique_id}+{3 + unique_id}? Answer in one word:"
    )
    return header + padding + footer


async def probe_single(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    timeout: float = 120.0,
) -> tuple[bool, str]:
    """Send one request and return (success, detail)."""
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            timeout=timeout,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        return (
            True,
            f"ok (prompt={prompt_tokens}, total={total_tokens}, reply={text!r})",
        )
    except TimeoutError:
        return False, "TIMEOUT (>120s)"
    except Exception as e:
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return False, f"ERROR: {msg}"


async def probe_concurrency(
    client: AsyncOpenAI,
    model: str,
    prompts: list[str],
    timeout: float = 180.0,
) -> tuple[bool, list[tuple[bool, str]]]:
    """Send len(prompts) simultaneous requests. All must succeed."""
    tasks = [probe_single(client, model, p, timeout=timeout) for p in prompts]
    results = await asyncio.gather(*tasks)
    all_ok = all(ok for ok, _ in results)
    return all_ok, list(results)


def _log(msg: str) -> None:
    """Log via print with [calibrate] prefix (works in orchestrator context)."""
    if _standalone:
        log.info(msg)
    else:
        print(f"[calibrate] {msg}", flush=True)


async def calibrate(
    url: str,
    target_tokens: int,
    max_concurrency: int,
    timeout_per_request: float,
) -> dict:
    """Run the calibration ladder: concurrency 1, 2, 3, ... until failure."""
    client = AsyncOpenAI(base_url=url, api_key="EMPTY")

    models = await client.models.list()
    model = models.data[0].id if models.data else "unknown"
    _log(f"Model: {model}")
    _log(f"Target tokens per request: {target_tokens}")
    _log(f"Max concurrency to test: {max_concurrency}")
    _log("")

    # Pre-build unique prompts to defeat prefix caching. Each prompt has
    # different content so vLLM can't share KV cache across requests.
    prompts = [
        build_padded_prompt(target_tokens, unique_id=i) for i in range(max_concurrency)
    ]
    est_tok = int(len(prompts[0]) / _CHARS_PER_TOKEN)
    _log(
        f"Built {len(prompts)} unique prompts, "
        f"~{len(prompts[0])} chars each (~{est_tok} tokens)"
    )
    _log("")

    last_passing = 0
    results_log: list[dict] = []

    for c in range(1, max_concurrency + 1):
        _log(f"--- Concurrency {c} ---")
        t0 = time.monotonic()
        all_ok, details = await probe_concurrency(
            client, model, prompts[:c], timeout=timeout_per_request
        )
        elapsed = time.monotonic() - t0

        for i, (ok, detail) in enumerate(details):
            status = "PASS" if ok else "FAIL"
            _log(f"  request {i + 1}/{c}: {status} — {detail}")

        passed = sum(1 for ok, _ in details if ok)
        failed = c - passed

        entry = {
            "concurrency": c,
            "passed": passed,
            "failed": failed,
            "all_ok": all_ok,
            "elapsed_s": round(elapsed, 1),
        }
        results_log.append(entry)

        if all_ok:
            last_passing = c
            _log(f"  → PASS (all {c} requests succeeded in {elapsed:.1f}s)")
        else:
            _log(f"  → FAIL ({failed}/{c} failed in {elapsed:.1f}s) — stopping")
            break

        _log("")

    _log("")
    _log("=" * 60)
    tok_k = target_tokens // 1000
    _log(
        f"CALIBRATION RESULT: max safe concurrency at {tok_k}K tokens = {last_passing}"
    )
    _log("=" * 60)

    return {
        "model": model,
        "target_tokens": target_tokens,
        "max_safe_concurrency": last_passing,
        "results": results_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        required=True,
        help="vLLM endpoint URL (e.g. http://localhost:26048/v1)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=100000,
        help="Target tokens per request (default: 100000)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=12,
        help="Maximum concurrency to test (default: 12)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout per request in seconds (default: 180)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write/update calibration JSON",
    )
    parser.add_argument(
        "--gpu-key",
        help="Key in calibration JSON to update (e.g. 'baobab/a100_80')",
    )
    args = parser.parse_args()

    result = asyncio.run(
        calibrate(args.url, args.tokens, args.max_concurrency, args.timeout)
    )

    if args.output and args.gpu_key:
        cal = {}
        if args.output.exists():
            cal = json.loads(args.output.read_text())

        cal.setdefault(args.gpu_key, {})
        cal[args.gpu_key]["calibrated_at_tokens"] = args.tokens
        cal[args.gpu_key]["calibrated_max_concurrency"] = result["max_safe_concurrency"]
        cal[args.gpu_key]["calibration_results"] = result["results"]
        cal[args.gpu_key]["calibrated_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(cal, indent=2, ensure_ascii=False))
        _log(f"Updated {args.output} [{args.gpu_key}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

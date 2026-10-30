"""Line-level integration tests — one-per-rule reachability checks.

Each test seeds the minimal ``PipelineState`` required to satisfy the
rule under test's preconditions (and every prior rule's preconditions,
nothing more), invokes the natural entry point, and asserts the rule
fires with the expected state transition.

A rule is "reachable" if its test passes. Skipped tests flag rules that
cannot be exercised via in-process construction — those require
integration-level fixtures or live infra.
"""

__all__: list[str] = []

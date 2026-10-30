"""In-pipeline primitives that run at submission / emission time.

Distinct from ``r2c2_agent.agent.submission_validator`` (which validates the
final PR/AC file content after it has been formatted and written). The
helpers here run inside the agentic pipeline — e.g. ``validate_r2c2_answer``
is called at every tier-emission site before the envelope is populated.
"""

__all__: list[str] = []

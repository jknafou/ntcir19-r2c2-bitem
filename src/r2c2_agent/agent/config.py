"""Configuration dataclasses for the Agent-BiTeM pipeline.

These are plain dataclasses — no external dependencies. They define
the runtime configuration for the agentic loop, service endpoints,
and run metadata.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from r2c2_agent.agent.loop_hooks import DraftContext, DraftDecision

__all__ = [
    "LoopConfig",
    "Mode",
    "NER_TOOLS",
    "RunConfig",
    "ServiceEndpoints",
]

# Tools that must be routed to the NER-enriched MCP service instance.
# Listed explicitly rather than matched by a name suffix.
NER_TOOLS: frozenset[str] = frozenset({"search_by_entity", "search_entities"})


class Mode(StrEnum):
    """Operational modes for the R2C2 pipeline."""

    HUMAN_GOLD = "a"
    STUDENT_AGENT = "b"
    HUMAN_JUDGE = "c"
    AUTO_JUDGE = "d"


@dataclass
class LoopConfig:
    """Configuration for the main agentic loop.

    Defaults match orchestrator-spec.md §3.2.
    """

    max_iterations: int = 10
    context_budget: int = 34816  # 85% of 40960 (in-loop compaction threshold)
    token_budget: int = (
        24576  # hard cap — reject trace if input token count exceeds this
    )
    # Last-resort guard against a wedged call, not a tuning knob. Bulk
    # runs bound wall-clock time per question at the orchestrator, so this
    # is generous (15 min) and slow but healthy MCP, NLI or vLLM calls
    # never trigger it.
    tool_timeout: float = 900.0  # seconds
    synthesis_timeout: float = 60.0  # seconds for forced synthesis
    concurrency: int = 1  # parallel questions (mode b only)
    # Optional verifier-driven self-correction callback. Called after
    # every draft answer by the ``answer_question`` driver only; other
    # drivers ignore it. ``None`` skips the hook. Forward-referenced so
    # importing ``config`` does not pull in ``loop_hooks``.
    on_draft: Callable[[DraftContext], Awaitable[DraftDecision | None]] | None = None
    # Free-form bag for adapter-specific extras without widening the
    # dataclass surface; not consumed by the loop itself.
    extras: dict[str, Any] = field(default_factory=dict)
    # Answer of an earlier rollout of the same ensemble, filled in by the
    # per-iteration worker when one exists. The pipeline carries these on
    # the PipelineState so force_deliver's final fallback can verify the
    # answer through answer_search_support before falling to tier-5.
    # The defaults leave single-rollout runs unaffected.
    prior_rollout_answer: str | None = None
    prior_rollout_confidence: int = 0
    prior_rollout_source_iter: int | None = None


@dataclass
class ServiceEndpoints:
    """URLs for external services.

    Two MCP endpoints because NER tools live on a separate service
    instance with the NER-enriched index.
    """

    main_model: str  # "http://host:port/v1" (122B)
    verifier: str | None = None  # "http://host:port/v1" (27B)
    mcp_standard: str = ""  # mcp-r2c2 (standard index)
    mcp_ner: str = ""  # mcp-r2c2-ner (NER-enriched index)


@dataclass
class RunConfig:
    """Metadata for a pipeline execution run."""

    run_name: str = "BITEM-PG-1"
    mode: Mode = Mode.STUDENT_AGENT
    topics_path: Path = field(default_factory=lambda: Path("topics.xml"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    dummy: bool = False
    endpoints: ServiceEndpoints = field(
        default_factory=lambda: ServiceEndpoints(main_model="")
    )
    loop: LoopConfig = field(default_factory=LoopConfig)

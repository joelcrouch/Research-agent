"""
agent/state.py

AgentState is the single shared data structure passed between every node
in the LangGraph state machine. All nodes read from and write to this dict.

Sprint 0: minimal fields — query, plan, tool_results, response, error, trace.
Sprint 1 will add: papers, top_k, date_from, date_to.
"""

from typing import Any

from typing_extensions import TypedDict


class TraceEntry(TypedDict):
    """One entry in the execution trace — written by each node."""

    node: str
    timestamp: str
    duration_ms: int
    input_summary: str
    output_summary: str
    error: str | None


class AgentState(TypedDict):
    """
    Shared state passed through every node of the LangGraph state machine.

    Fields are intentionally typed strictly — use `None` for optional values,
    never omit keys, so downstream nodes can always access state["field"]
    without KeyError.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    query: str  # raw user query, e.g. "long-term potentiation"

    # ── Planner output ─────────────────────────────────────────────────────
    plan: list[str]  # search terms extracted/expanded by the planner

    # ── Tool outputs ───────────────────────────────────────────────────────
    tool_results: list[dict[str, Any]]  # raw tool call results (Sprint 0: empty list)

    # ── Final output ───────────────────────────────────────────────────────
    final_response: str | None  # formatted response string returned to the user

    # ── Error handling ─────────────────────────────────────────────────────
    error: str | None  # set if any node fails; triggers early exit

    # ── Observability ──────────────────────────────────────────────────────
    trace_id: str  # unique ID for this run (UUID4)
    trace: list[TraceEntry]  # execution trace — appended to by each node

import time
from datetime import UTC, datetime

import structlog

from agent.state import AgentState, TraceEntry

log = structlog.get_logger(__name__)


def _make_trace_entry(
    node: str,
    start_time: float,
    input_summary: str,
    output_summary: str,
    error: str | None = None,
) -> TraceEntry:
    duration_ms = int((time.monotonic() - start_time) * 1000)
    return TraceEntry(
        node=node,
        timestamp=datetime.now(UTC).isoformat(),
        duration_ms=duration_ms,
        input_summary=input_summary,
        output_summary=output_summary,
        error=error,
    )


def planner(state: AgentState) -> dict:
    start = time.monotonic()
    node_name = "planner"
    log.info(f"{node_name}.enter", trace_id=state["trace_id"], query=state["query"])

    # Stub: plan is just the raw query wrapped in a list
    plan = [state["query"]]

    entry = _make_trace_entry(
        node=node_name,
        start_time=start,
        input_summary=f"query='{state['query']}'",
        output_summary=f"plan={plan}",
    )

    log.info(
        f"{node_name}.exit", trace_id=state["trace_id"], plan=plan, duration_ms=entry["duration_ms"]
    )

    return {
        "plan": plan,
        "trace": state["trace"] + [entry],
    }


# ── Node 2: Tool Caller ────────────────────────────────────────────────────────
def tool_caller(state: AgentState) -> dict:  # type: ignore[type-arg]
    """
    Sprint 0 stub: no-op.
    Sprint 1 will replace this with a ReAct LangGraph tool-calling node
    that invokes search_arxiv and get_citation_count.
    """
    start = time.monotonic()
    node_name = "tool_caller"

    log.info(f"{node_name}.enter", trace_id=state["trace_id"], plan=state["plan"])

    # Stub: no tools called yet
    tool_results: list = []

    entry = _make_trace_entry(
        node=node_name,
        start_time=start,
        input_summary=f"plan={state['plan']}",
        output_summary="tool_results=[] (stub)",
    )

    log.info(
        f"{node_name}.exit",
        trace_id=state["trace_id"],
        result_count=0,
        duration_ms=entry["duration_ms"],
    )

    return {
        "tool_results": tool_results,
        "trace": state["trace"] + [entry],
    }


# ── Node 3: Responder ──────────────────────────────────────────────────────────
def responder(state: AgentState) -> dict:  # type: ignore[type-arg]
    """
    Sprint 0 stub: returns a hardcoded placeholder response.
    Sprint 1 will replace this with real deduplication, ranking, and formatting.
    """
    start = time.monotonic()
    node_name = "responder"

    log.info(f"{node_name}.enter", trace_id=state["trace_id"])

    if state.get("error"):
        response = f"[ERROR] Agent encountered an error: {state['error']}"
    else:
        # Stub response — real formatting added in Sprint 1
        response = (
            f"[STUB] Query received: '{state['query']}'\n"
            f"Plan: {state['plan']}\n"
            f"Tool results: {len(state['tool_results'])} (none yet — Sprint 0 stub)\n"
            "Real paper results will appear here after Sprint 1."
        )

    entry = _make_trace_entry(
        node=node_name,
        start_time=start,
        input_summary=f"tool_results_count={len(state['tool_results'])}",
        output_summary=f"response_length={len(response)}",
    )

    log.info(f"{node_name}.exit", trace_id=state["trace_id"], duration_ms=entry["duration_ms"])

    return {
        "final_response": response,
        "trace": state["trace"] + [entry],
    }

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog
from langgraph.graph import END, START, StateGraph

from agent.nodes import planner, responder, tool_caller
from agent.state import AgentState

log = structlog.get_logger(__name__)


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("tool_caller", tool_caller)
    graph.add_node("responder", responder)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tool_caller")
    graph.add_edge("tool_caller", "responder")
    graph.add_edge("responder", END)

    return graph


# ompliel one at import time
_compiled = _build_graph().compile()


def run(query: str) -> AgentState:
    trace_id = str(uuid.uuid4())
    log.info("agent.run.start", trace_id=trace_id, query=query)

    initial_state: AgentState = {
        "query": query,
        "plan": [],
        "tool_results": [],
        "final_response": None,
        "error": None,
        "trace_id": trace_id,
        "trace": [],
    }

    try:
        final_state: AgentState = _compiled.invoke(initial_state)
    except Exception as exc:
        log.exception("agent.run.error", trace_id=trace_id, error=str(exc))
        final_state = {**initial_state, "error": str(exc)}

    _write_trace(trace_id, final_state)

    log.info(
        "agent.run.complete",
        trace_id=trace_id,
        node_count=len(final_state["trace"]),
        has_error=bool(final_state.get("error")),
    )

    return final_state


def _write_trace(trace_id: str, state: AgentState) -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    trace_path = logs_dir / f"trace_{trace_id}.json"
    payload = {
        "trace_id": trace_id,
        "query": state["query"],
        "timestamp": datetime.now(UTC).isoformat(),
        "error": state.get("error"),
        "nodes": state["trace"],
    }

    with trace_path.open("w") as f:
        json.dump(payload, f, indent=2)

    log.info("trace.written", path=str(trace_path))

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import structlog
from langgraph.graph import END, START, StateGraph

from agent.nodes import call_model, planner, responder, should_continue, tool_node
from agent.state import AgentState
from config.settings import get_settings

log = structlog.get_logger(__name__)


def _build_graph() -> StateGraph[AgentState]:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("planner", planner)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("responder", responder)

    # Edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")

    # Conditional ReAct Loop
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "responder": "responder",
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("responder", END)

    return graph


# Compile
_compiled = _build_graph().compile()


def run(query: str) -> AgentState:
    trace_id = str(uuid.uuid4())
    settings = get_settings()

    # Default date range from settings
    now = datetime.now(UTC)
    date_to = now.strftime("%Y-%m-%d")
    date_from = (now - timedelta(days=settings.default_date_range_years * 365)).strftime("%Y-%m-%d")

    log.info("agent.run.start", trace_id=trace_id, query=query)

    initial_state: AgentState = {
        "query": query,
        "messages": [],  # Added for agentic loop
        "plan": [],
        "top_k": settings.default_top_k,
        "date_from": date_from,
        "date_to": date_to,
        "tool_results": [],
        "papers": [],
        "final_response": None,
        "error": None,
        "trace_id": trace_id,
        "trace": [],
    }

    try:
        # LangGraph invoke sometimes has complex type requirements for TypedDicts
        final_state = cast(AgentState, _compiled.invoke(initial_state))  # type: ignore[arg-type]
    except Exception as exc:
        log.exception("agent.run.error", trace_id=trace_id, error=str(exc))
        # Create a final state that looks like AgentState
        error_state = initial_state.copy()
        error_state["error"] = str(exc)
        final_state = error_state

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

    # Convert messages to serializable format
    serializable_messages = []
    for msg in state.get("messages", []):
        serializable_messages.append(
            {
                "type": msg.type,
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", None),
            }
        )

    payload = {
        "trace_id": trace_id,
        "query": state["query"],
        "timestamp": datetime.now(UTC).isoformat(),
        "error": state.get("error"),
        "nodes": state["trace"],
        "messages": serializable_messages,
    }

    with trace_path.open("w") as f:
        json.dump(payload, f, indent=2)

    log.info("trace.written", path=str(trace_path))

import json
import time
from datetime import UTC, datetime
from typing import Any, Literal, cast

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.llm import get_llm
from agent.state import AgentState, TraceEntry
from agent.tools import TOOLS, _get_citation_count, _search_arxiv, deduplicate, rank_papers
from schemas.paper import Paper

log = structlog.get_logger(__name__)


class PlannerOutput(BaseModel):
    """Structured plan from the LLM."""

    search_terms: list[str] = Field(description="Core search terms.")
    date_from: str = Field(description="Start date (YYYY-MM-DD).")
    date_to: str = Field(description="End date (YYYY-MM-DD).")
    top_k: int = Field(description="Number of papers.")


def _make_trace_entry(
    node: str, start_time: float, input_summary: str, output_summary: str, error: str | None = None
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


# ── Node 1: Planner ───────────────────────────────────────────────────────────
def planner(state: AgentState) -> dict[str, Any]:
    start = time.monotonic()
    log.info("planner.enter", query=state["query"])

    try:
        llm = get_llm().with_structured_output(PlannerOutput)
        system_prompt = (
            "You are a research query planner. Decompose the query into search terms and a date range. "
            f"The current date is {datetime.now(UTC).strftime('%Y-%m-%d')}. "
            f"Defaults to use if not specified: date_from='{state['date_from']}', "
            f"date_to='{state['date_to']}', top_k={state['top_k']}."
        )

        response = llm.invoke([("system", system_prompt), ("human", state["query"])])

        # Ensure we have a PlannerOutput instance
        if isinstance(response, dict):
            plan_out = PlannerOutput(**response)
        else:
            plan_out = cast(PlannerOutput, response)

        # Initialize messages for the agentic loop
        initial_msg = HumanMessage(
            content=(
                f"Plan: {plan_out.search_terms}. "
                f"Range: {plan_out.date_from} to {plan_out.date_to}. "
                f"Goal: Find and enrich top {plan_out.top_k} papers."
            )
        )

        entry = _make_trace_entry("planner", start, state["query"], f"plan={plan_out.search_terms}")
        return {
            "plan": plan_out.search_terms,
            "date_from": plan_out.date_from,
            "date_to": plan_out.date_to,
            "top_k": plan_out.top_k,
            "messages": [initial_msg],
            "trace": state["trace"] + [entry],
        }
    except Exception as e:
        log.error("planner.error", error=str(e))
        return {"error": f"Planner failed: {str(e)}"}


# ── Node 2: Agent (Model Caller) ──────────────────────────────────────────────
def call_model(state: AgentState) -> dict[str, Any]:
    start = time.monotonic()
    log.info("agent.call_model", messages_count=len(state["messages"]))

    # We cast to any because LangChain's bind_tools might have complex return types
    llm = cast(Any, get_llm()).bind_tools(TOOLS)
    system_msg = SystemMessage(
        content=(
            "You are a research agent. Use 'search_arxiv' to find papers and 'get_citation_count' to enrich them. "
            "Search for ALL terms in the plan. Enrich the most relevant papers found. "
            "When you have enough papers with citation counts, provide a final summary and stop."
        )
    )

    # LangChain messages concatenation requires a list of BaseMessage
    messages: list[BaseMessage] = [cast(BaseMessage, system_msg)] + list(state["messages"])
    response = llm.invoke(messages)

    # Check for tool calls on the response
    has_tool_calls = hasattr(response, "tool_calls") and bool(response.tool_calls)

    entry = _make_trace_entry(
        "agent", start, "LLM decision", f"type={'tool_call' if has_tool_calls else 'final'}"
    )
    return {"messages": [response], "trace": state["trace"] + [entry]}


# ── Node 3: Tool Node ─────────────────────────────────────────────────────────
def tool_node(state: AgentState) -> dict[str, Any]:
    """Executes tool calls and updates the papers list."""
    start = time.monotonic()
    last_msg = cast(AIMessage, state["messages"][-1])

    new_messages = []
    found_papers: list[Paper] = list(state.get("papers", []))

    if hasattr(last_msg, "tool_calls"):
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            log.info("tool.executing", tool=tool_name, args=args)

            # Execute the actual logic and also update state["papers"]
            if tool_name == "search_arxiv":
                papers = _search_arxiv(**args)
                found_papers.extend(papers)
                content = f"Found {len(papers)} papers for query '{args.get('query')}'."
            elif tool_name == "get_citation_count":
                p_stub = Paper(
                    title=args["title"],
                    author=[],
                    abstract=None,
                    year=None,
                    arxiv_id=args["arxiv_id"],
                    pubmed_id=None,
                    doi=None,
                    semantic_scholar_id=None,
                    citation_count=None,
                    source="arxiv",
                    url="",
                )
                enriched = _get_citation_count(p_stub)

                # Update our internal paper list if title matches
                for p in found_papers:
                    if p.title.lower() == enriched.title.lower() or p.arxiv_id == enriched.arxiv_id:
                        p.citation_count = enriched.citation_count
                        p.semantic_scholar_id = enriched.semantic_scholar_id

                content = f"Citations for '{args['title']}': {enriched.citation_count}"
            else:
                content = f"Error: Tool {tool_name} not found."

            new_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

    entry = _make_trace_entry(
        "tools", start, f"calls={len(getattr(last_msg, 'tool_calls', []))}", "executed"
    )
    return {"messages": new_messages, "papers": found_papers, "trace": state["trace"] + [entry]}


# ── Node 4: Responder ──────────────────────────────────────────────────────────
def responder(state: AgentState) -> dict[str, Any]:
    import io

    from rich.console import Console
    from rich.table import Table

    start = time.monotonic()
    log.info("responder.enter")

    # Final cleanup
    final_papers = deduplicate(state["papers"])
    ranked_papers = rank_papers(final_papers, state["top_k"])

    output_buffer = io.StringIO()
    console = Console(file=output_buffer, force_terminal=True, width=120)
    table = Table(title=f"Research Results: {state['query']}")
    table.add_column("Rank", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Year", style="green")
    table.add_column("Citations", style="magenta")
    table.add_column("URL", style="blue")

    for i, p in enumerate(ranked_papers, 1):
        table.add_row(
            str(i),
            (p.title[:57] + "...") if len(p.title) > 60 else p.title,
            str(p.year or "N/A"),
            str(p.citation_count or "N/A"),
            p.url,
        )

    console.print(table)
    response = output_buffer.getvalue()

    # Sprint 1 Requirement: Write results to JSON
    run_id = state["trace_id"]
    res_path = f"logs/result_{run_id}.json"
    try:
        with open(res_path, "w") as f:
            json.dump([p.__dict__ for p in ranked_papers], f, indent=2)
    except Exception as e:
        log.error("responder.json_write_error", path=res_path, error=str(e))

    entry = _make_trace_entry("responder", start, f"papers={len(ranked_papers)}", "done")
    return {"final_response": response, "trace": state["trace"] + [entry]}


def should_continue(state: AgentState) -> Literal["tools", "responder"]:
    """Conditional edge to decide if we loop or finish."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and cast(Any, last_message).tool_calls:
        # Max iterations guard (Sprint 1: 10 calls)
        if len(state["messages"]) > 10:
            log.warning("agent.max_iterations_hit")
            return "responder"
        return "tools"
    return "responder"

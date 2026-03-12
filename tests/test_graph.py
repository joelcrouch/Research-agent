"""
tests/test_graph.py

Sprint 1 smoke tests.
Verifies the agentic ReAct graph compiles and runs.
"""

import pytest
from unittest.mock import MagicMock, patch
from agent.graph import run
from agent.nodes import PlannerOutput
from schemas.paper import Paper
from langchain_core.messages import AIMessage

@pytest.fixture(autouse=True)
def mock_external_deps():
    """Globally mock LLM and tools for graph tests."""
    with patch("agent.nodes.get_llm") as mock_get_llm, \
         patch("agent.nodes._search_arxiv") as mock_search, \
         patch("agent.nodes._get_citation_count") as mock_enrich:
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Planner mock
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = PlannerOutput(
            search_terms=["test term"],
            date_from="2020-01-01",
            date_to="2025-01-01",
            top_k=5
        )
        
        # Agent mock: Use REAL AIMessage objects, not mocks
        msg_with_tools = AIMessage(
            content="", 
            tool_calls=[{
                "name": "search_arxiv",
                "args": {"query": "test term", "max_results": 5, "date_from": "2020-01-01", "date_to": "2025-01-01"},
                "id": "call_1"
            }]
        )
        msg_final = AIMessage(content="I have found the papers.")
        
        # Crucial: bind_tools must return the mock_llm itself
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [msg_with_tools, msg_final]
        
        # Mock ArXiv
        mock_search.return_value = [
            Paper(title="Test Paper", author=[], abstract=None, year=2020, arxiv_id="1", 
                  pubmed_id=None, doi=None, semantic_scholar_id=None, 
                  citation_count=None, source="arxiv", url="url")
        ]
        
        # Mock enrichment
        def side_effect(p):
            p.citation_count = 10
            return p
        mock_enrich.side_effect = side_effect
        
        yield

class TestGraphSmoke:
    def test_run_returns_agent_state(self):
        state = run("test query")
        assert isinstance(state, dict)
        assert "messages" in state
        # HumanMessage (from planner) + AIMessage (tools) + ToolMessage (from tool_node) + AIMessage (final)
        assert len(state["messages"]) >= 4
        assert "final_response" in state
        assert "papers" in state

    def test_trace_has_correct_nodes(self):
        state = run("test query")
        node_names = [entry["node"] for entry in state["trace"]]
        assert "planner" in node_names
        assert "agent" in node_names
        assert "tools" in node_names
        assert "responder" in node_names

    def test_result_json_is_written(self, tmp_path, monkeypatch):
        # Create logs dir in tmp_path
        (tmp_path / "logs").mkdir()
        monkeypatch.chdir(tmp_path)
        state = run("test query")
        res_file = tmp_path / "logs" / f"result_{state['trace_id']}.json"
        assert res_file.exists()

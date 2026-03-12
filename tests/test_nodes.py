import pytest
from unittest.mock import MagicMock, patch
from agent.nodes import planner, call_model, tool_node, responder, PlannerOutput
from agent.state import AgentState
from schemas.paper import Paper
from langchain_core.messages import AIMessage, ToolMessage

@pytest.fixture
def base_state() -> AgentState:
    return {
        "query": "test query",
        "messages": [],
        "plan": [],
        "top_k": 5,
        "date_from": "2020-01-01",
        "date_to": "2025-01-01",
        "tool_results": [],
        "papers": [],
        "final_response": None,
        "error": None,
        "trace_id": "test-trace-id",
        "trace": [],
    }

class TestPlannerNode:
    @patch("agent.nodes.get_llm")
    def test_planner_success(self, mock_get_llm, base_state):
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        mock_structured_llm.invoke.return_value = PlannerOutput(
            search_terms=["term1"],
            date_from="2021-01-01",
            date_to="2024-01-01",
            top_k=10
        )
        
        result = planner(base_state)
        assert result["plan"] == ["term1"]
        assert len(result["messages"]) == 1

class TestAgentNodes:
    @patch("agent.nodes.get_llm")
    def test_call_model_tool_choice(self, mock_get_llm, base_state):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm
        
        mock_llm.invoke.return_value = AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "test"}, "id": "1"}])
        
        result = call_model(base_state)
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "search_arxiv"

    @patch("agent.nodes._search_arxiv")
    def test_tool_node_execution(self, mock_search, base_state):
        base_state["messages"] = [AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "test"}, "id": "1"}])]
        mock_search.return_value = [Paper(title="P1", author=[], abstract=None, year=2020, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="arxiv", url="url")]
        
        result = tool_node(base_state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert len(result["papers"]) == 1

class TestResponderNode:
    def test_responder_output(self, base_state):
        base_state["papers"] = [Paper(title="P1", author=[], abstract=None, year=2020, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=10, source="arxiv", url="url")]
        result = responder(base_state)
        assert "P1" in result["final_response"]
        assert "10" in result["final_response"]

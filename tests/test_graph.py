"""
tests/test_graph.py

Sprint 0 smoke tests.
Verifies the graph compiles, runs, and produces correctly shaped output.
No real API calls — all LLM interactions are stubbed via the mock_settings fixture.
"""

from agent.graph import run


class TestGraphSmoke:
    """Basic smoke tests — graph compiles and runs without errors."""

    def test_run_returns_agent_state(self):
        """run() should return a dict with all AgentState keys."""
        state = run("long-term potentiation")

        assert isinstance(state, dict)
        # All required keys must be present
        required_keys = {
            "query",
            "plan",
            "tool_results",
            "final_response",
            "error",
            "trace_id",
            "trace",
        }
        assert required_keys.issubset(state.keys())

    def test_query_is_preserved_in_state(self):
        """The original query should be unchanged in the final state."""
        query = "CRISPR gene editing"
        state = run(query)
        assert state["query"] == query

    def test_trace_id_is_set(self):
        """Every run must have a non-empty trace_id."""
        state = run("test query")
        assert state["trace_id"]
        assert len(state["trace_id"]) == 36  # UUID4 format

    def test_trace_has_all_three_nodes(self):
        """Trace should contain an entry for planner, tool_caller, and responder."""
        state = run("test query")
        node_names = [entry["node"] for entry in state["trace"]]
        assert "planner" in node_names
        assert "tool_caller" in node_names
        assert "responder" in node_names

    def test_no_error_on_valid_query(self):
        """A normal query should produce no error."""
        state = run("long-term potentiation")
        assert state["error"] is None

    def test_final_response_is_string(self):
        """final_response should be a non-empty string on success."""
        state = run("test query")
        assert isinstance(state["final_response"], str)
        assert len(state["final_response"]) > 0

    def test_plan_is_list(self):
        """plan should be a list (even if it only contains the raw query in Sprint 0)."""
        state = run("test query")
        assert isinstance(state["plan"], list)
        assert len(state["plan"]) >= 1


class TestTraceSchema:
    """Verify the trace entry schema is correct."""

    def test_trace_entries_have_required_fields(self):
        """Every trace entry must have the required fields."""
        state = run("test query")
        required = {"node", "timestamp", "duration_ms", "input_summary", "output_summary", "error"}
        for entry in state["trace"]:
            assert required.issubset(entry.keys()), f"Trace entry missing fields: {entry}"

    def test_trace_duration_is_non_negative(self):
        """Duration must be a non-negative integer."""
        state = run("test query")
        for entry in state["trace"]:
            assert isinstance(entry["duration_ms"], int)
            assert entry["duration_ms"] >= 0

    def test_trace_error_is_none_on_success(self):
        """On a successful run, all trace entry errors should be None."""
        state = run("test query")
        for entry in state["trace"]:
            assert entry["error"] is None


class TestTraceFile:
    """Verify the trace JSON file is written correctly."""

    def test_trace_file_is_written(self, tmp_path, monkeypatch):
        """run() should write a trace file to the logs/ directory."""
        # Redirect logs to tmp_path so we don't pollute the project dir
        monkeypatch.chdir(tmp_path)
        state = run("test query")

        trace_file = tmp_path / "logs" / f"trace_{state['trace_id']}.json"
        assert trace_file.exists(), f"Expected trace file at {trace_file}"

    def test_trace_file_is_valid_json(self, tmp_path, monkeypatch):
        """The trace file should be valid JSON with expected top-level keys."""
        import json

        monkeypatch.chdir(tmp_path)
        state = run("test query")

        trace_file = tmp_path / "logs" / f"trace_{state['trace_id']}.json"
        with trace_file.open() as f:
            data = json.load(f)

        assert data["trace_id"] == state["trace_id"]
        assert data["query"] == state["query"]
        assert "nodes" in data
        assert isinstance(data["nodes"], list)

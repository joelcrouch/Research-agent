import pytest
import os
from agent.graph import run

@pytest.mark.integration
def test_sprint1_end_to_end_crispr():
    """
    S1-10: Integration Test
    Query 'CRISPR gene editing', assert at least 5 papers, 
    and verify citation-based ranking.
    """
    # Verify environment has keys if needed (Ollama doesn't need them)
    # but we assume the environment is set up.
    
    query = "CRISPR gene editing"
    state = run(query)
    
    assert state["error"] is None
    assert "final_response" in state
    
    papers = state["papers"]
    assert len(papers) >= 5
    
    # Verify citation ranking (descending)
    # Note: Some papers might have None citation count, they go to bottom.
    counts = [p.citation_count for p in papers if p.citation_count is not None]
    assert counts == sorted(counts, reverse=True)
    
    # Verify results JSON was written
    run_id = state["trace_id"]
    assert os.path.exists(f"logs/result_{run_id}.json")
    print(f"\nIntegration Success! Found {len(papers)} papers.")

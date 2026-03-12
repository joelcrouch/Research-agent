import pytest
from schemas.paper import Paper
from agent.tools.ranker import rank_papers

def mk_paper(citations):
    return Paper(
        title="T", author=[], abstract=None, year=None, arxiv_id=None, pubmed_id=None, doi=None, semantic_scholar_id=None, 
        citation_count=citations, source="", url=""
    )

def test_rank_empty():
    assert rank_papers([], 5) == []

def test_rank_simple_descending():
    p1 = mk_paper(10)
    p2 = mk_paper(50)
    p3 = mk_paper(5)
    
    ranked = rank_papers([p1, p2, p3], 3)
    assert [p.citation_count for p in ranked] == [50, 10, 5]

def test_rank_with_nones():
    p1 = mk_paper(10)
    p2 = mk_paper(None) # Should be last
    p3 = mk_paper(100)
    
    ranked = rank_papers([p1, p2, p3], 3)
    assert [p.citation_count for p in ranked] == [100, 10, None]

def test_rank_top_k_limit():
    papers = [mk_paper(i) for i in range(10)] # 0..9
    # Should return [9, 8, 7, 6, 5]
    ranked = rank_papers(papers, 5)
    assert len(ranked) == 5
    assert ranked[0].citation_count == 9
    assert ranked[-1].citation_count == 5

def test_rank_stable_nones():
    # If multiple Nones, original order should be preserved (stability)
    p1 = mk_paper(None); p1.title = "A"
    p2 = mk_paper(None); p2.title = "B"
    
    ranked = rank_papers([p1, p2], 2)
    assert ranked[0].title == "A"
    assert ranked[1].title == "B"

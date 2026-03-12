import pytest
from schemas.paper import Paper
from agent.tools.deduplicator import deduplicate

@pytest.fixture
def base_paper():
    return Paper(
        title="Attention is All You Need",
        author=["Vaswani et al"],
        abstract="Transformers...",
        year=2017,
        arxiv_id="1706.03762",
        pubmed_id=None,
        doi="10.48550/arXiv.1706.03762",
        semantic_scholar_id="ss_1",
        citation_count=100,
        source="arxiv",
        url="http://arxiv.org"
    )

def test_deduplicate_empty_list():
    assert deduplicate([]) == []

def test_deduplicate_no_duplicates(base_paper):
    paper2 = Paper(
        title="BERT",
        author=["Devlin et al"],
        abstract="Bidirectional...",
        year=2018,
        arxiv_id="1810.04805",
        pubmed_id=None,
        doi="10.48550/arXiv.1810.04805",
        semantic_scholar_id="ss_2",
        citation_count=50,
        source="arxiv",
        url="http://arxiv.org/bert"
    )
    results = deduplicate([base_paper, paper2])
    assert len(results) == 2

def test_deduplicate_by_doi(base_paper):
    # Same DOI, different title format
    duplicate = Paper(
        title="Attention Is All You Need (Updated)",
        author=[],
        abstract=None,
        year=None,
        arxiv_id=None,
        pubmed_id=None,
        doi=base_paper.doi, # Same DOI
        semantic_scholar_id=None,
        citation_count=None,
        source="other",
        url=""
    )
    
    results = deduplicate([base_paper, duplicate])
    assert len(results) == 1
    assert results[0].doi == base_paper.doi
    # Should keep original title as it was first
    assert results[0].title == base_paper.title

def test_deduplicate_by_arxiv_id(base_paper):
    # Same ArXiv ID, no DOI on second one
    duplicate = Paper(
        title="Attention Is All You Need",
        author=[],
        abstract=None,
        year=None,
        arxiv_id=base_paper.arxiv_id, # Same ArXiv ID
        pubmed_id=None,
        doi=None,
        semantic_scholar_id=None,
        citation_count=None,
        source="other",
        url=""
    )
    
    results = deduplicate([base_paper, duplicate])
    assert len(results) == 1
    assert results[0].arxiv_id == base_paper.arxiv_id

def test_deduplicate_by_title_fuzzy(base_paper):
    # Different ID types (simulate cross-source), but same title
    duplicate = Paper(
        title="attention is all you need.", # Lowercase + period
        author=[],
        abstract=None,
        year=2017,
        arxiv_id=None,
        pubmed_id="pm_123", # Has a different ID type
        doi=None,
        semantic_scholar_id=None,
        citation_count=None,
        source="pubmed",
        url=""
    )
    
    results = deduplicate([base_paper, duplicate])
    assert len(results) == 1
    # Should merge the Pubmed ID into the result
    assert results[0].pubmed_id == "pm_123"

def test_merge_logic_completeness():
    # Paper A has Title, Year
    p1 = Paper(
        title="Test Paper",
        author=[],
        abstract=None,
        year=2024,
        arxiv_id=None,
        pubmed_id=None,
        doi="10.1000/test",
        semantic_scholar_id=None,
        citation_count=10,
        source="s1",
        url=""
    )
    # Paper B has Abstract, URL, Authors, Higher Citations
    p2 = Paper(
        title="Test Paper",
        author=["Alice", "Bob"],
        abstract="Full abstract here.",
        year=None,
        arxiv_id=None,
        pubmed_id=None,
        doi="10.1000/test", # Matches p1
        semantic_scholar_id=None,
        citation_count=20,
        source="s2",
        url="http://url"
    )
    
    results = deduplicate([p1, p2])
    assert len(results) == 1
    merged = results[0]
    
    assert merged.year == 2024      # Kept from p1
    assert merged.abstract == "Full abstract here." # Filled from p2
    assert merged.author == ["Alice", "Bob"]        # Filled from p2
    assert merged.citation_count == 20 # Max taken
    assert merged.url == "http://url"  # Filled from p2

def test_deduplicate_maintain_order():
    p1 = Paper(title="Paper A", author=[], abstract=None, year=2020, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    p2 = Paper(title="Paper B", author=[], abstract=None, year=2021, arxiv_id="2", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    p3 = Paper(title="Paper A", author=[], abstract=None, year=2020, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    
    results = deduplicate([p1, p2, p3])
    assert len(results) == 2
    assert results[0].title == "Paper A"
    assert results[1].title == "Paper B"

def test_normalize_title_empty():
    """Test that empty titles are handled safely."""
    p1 = Paper(title="", author=[], abstract=None, year=None, arxiv_id=None, pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    # Should be treated as distinct if it has no ID and no title (though realistic papers have titles)
    results = deduplicate([p1])
    assert len(results) == 1

def test_merge_preserves_target_data():
    """Verify that we don't overwrite existing target data with None from source."""
    p1 = Paper(
        title="Paper", author=["A"], abstract="Abstract", year=2020, 
        arxiv_id="1", pubmed_id="pm1", doi="d1", semantic_scholar_id="s1", 
        citation_count=10, source="src", url="url"
    )
    # Source has mostly None
    p2 = Paper(
        title="Paper", author=None, abstract=None, year=None, 
        arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, 
        citation_count=None, source="src", url=None
    )
    
    results = deduplicate([p1, p2])
    assert len(results) == 1
    merged = results[0]
    
    assert merged.abstract == "Abstract"
    assert merged.year == 2020
    assert merged.pubmed_id == "pm1"
    assert merged.citation_count == 10
    assert merged.author == ["A"]

def test_merge_authors_prefer_target_if_longer():
    """Verify author merging logic prefers the longer list."""
    p1 = Paper(
        title="Paper", author=["Alice", "Bob", "Charlie"], 
        abstract=None, year=None, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url=""
    )
    p2 = Paper(
        title="Paper", author=["Alice et al"], 
        abstract=None, year=None, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url=""
    )
    
    # p1 is target, p2 is source. p1 has longer author list.
    results = deduplicate([p1, p2])
    assert len(results) == 1
    assert results[0].author == ["Alice", "Bob", "Charlie"]

def test_merge_ids_from_source():
    """Verify that if target is missing IDs, it adopts them from source."""
    # Target has ArXiv ID only
    p1 = Paper(
        title="Paper", author=[], abstract=None, year=None, 
        arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, 
        citation_count=None, source="src", url=None
    )
    # Source has DOI, PubMed, SS, URL
    p2 = Paper(
        title="Paper", author=[], abstract=None, year=None, 
        arxiv_id="1", pubmed_id="pm1", doi="d1", semantic_scholar_id="s1", 
        citation_count=None, source="src", url="url1"
    )
    
    # p1 is target (first), p2 is source
    results = deduplicate([p1, p2])
    assert len(results) == 1
    merged = results[0]
    
    assert merged.doi == "d1"
    assert merged.pubmed_id == "pm1"
    assert merged.semantic_scholar_id == "s1"
    assert merged.url == "url1"

def test_merge_authors_prefer_source_if_longer():
    """Verify author merging logic prefers source if it is longer."""
    p1 = Paper(
        title="Paper", author=["Alice et al"], 
        abstract=None, year=None, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url=""
    )
    p2 = Paper(
        title="Paper", author=["Alice", "Bob"], 
        abstract=None, year=None, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url=""
    )
    
    # p1 is target, p2 is source. p2 has longer author list.
    results = deduplicate([p1, p2])
    assert len(results) == 1
    assert results[0].author == ["Alice", "Bob"]

def test_merge_ids_all_permutations():
    """Ensure all branches of ID merging are exercised."""
    # Case 1: Target has ID, Source does not
    p1 = Paper(title="T", author=[], abstract=None, year=None, arxiv_id="1", pubmed_id="pm1", doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    p2 = Paper(title="T", author=[], abstract=None, year=None, arxiv_id="1", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    
    res1 = deduplicate([p1, p2])
    assert res1[0].pubmed_id == "pm1"

    # Case 2: Target None, Source None
    p3 = Paper(title="U", author=[], abstract=None, year=None, arxiv_id="2", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    p4 = Paper(title="U", author=[], abstract=None, year=None, arxiv_id="2", pubmed_id=None, doi=None, semantic_scholar_id=None, citation_count=None, source="", url="")
    
    res2 = deduplicate([p3, p4])
    assert res2[0].pubmed_id is None

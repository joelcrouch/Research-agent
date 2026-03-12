from langchain_core.tools import tool

from agent.tools.arxiv_search import search_arxiv as _search_arxiv
from agent.tools.deduplicator import deduplicate
from agent.tools.ranker import rank_papers
from agent.tools.semantic_scholar import get_citation_count as _get_citation_count
from schemas.paper import Paper


@tool
def search_arxiv(query: str, max_results: int, date_from: str, date_to: str) -> str:
    """
    Search ArXiv for academic papers.
    Returns a list of papers with titles, authors, and ArXiv IDs.
    """
    papers = _search_arxiv(query, max_results, date_from, date_to)
    if not papers:
        return "No papers found for this query."

    # Return a summary for the LLM to read
    results = []
    for p in papers:
        results.append(
            f"Title: {p.title}\n"
            f"ArXiv ID: {p.arxiv_id}\n"
            f"Authors: {', '.join(p.author)}\n"
            f"URL: {p.url}\n"
            "---"
        )
    return "\n".join(results)


@tool
def get_citation_count(title: str, arxiv_id: str) -> str:
    """
    Get the citation count for a specific paper from Semantic Scholar.
    Requires the exact title and the ArXiv ID (e.g. '1706.03762').
    """
    # Create a minimal paper object for the tool to work with
    p = Paper(
        title=title,
        author=[],
        abstract=None,
        year=None,
        arxiv_id=arxiv_id,
        pubmed_id=None,
        doi=None,
        semantic_scholar_id=None,
        citation_count=None,
        source="arxiv",
        url="",
    )
    enriched = _get_citation_count(p)

    if enriched.citation_count is not None:
        return f"Citation count for '{title}': {enriched.citation_count}"
    return f"Could not find citation data for '{title}'."


# Map for the agent
TOOLS = [search_arxiv, get_citation_count]
__all__ = [
    "search_arxiv",
    "get_citation_count",
    "deduplicate",
    "rank_papers",
    "TOOLS",
    "_search_arxiv",
    "_get_citation_count",
]

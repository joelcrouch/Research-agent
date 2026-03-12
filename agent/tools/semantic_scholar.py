import httpx
import structlog
from typing import Any, cast

from schemas.paper import Paper

log = structlog.get_logger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"


def get_citation_count(paper: Paper) -> Paper:
    """
    Enrich a Paper object with citation count from Semantic Scholar.
    Lookup order: DOI -> ArXiv ID -> Title search.

    Args:
        paper: The Paper object to enrich.

    Returns:
        The updated Paper object with citation_count and semantic_scholar_id set if found.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            # 1. Try DOI
            if paper.doi:
                result = _get_by_id(client, f"DOI:{paper.doi}")
                if result:
                    return _update_paper(paper, result, "doi")

            # 2. Try ArXiv ID
            if paper.arxiv_id:
                # Semantic Scholar sometimes prefers ArXiv: or just the ID
                # We try arXiv: first as it is standard
                for prefix in ["arXiv:", ""]:
                    result = _get_by_id(client, f"{prefix}{paper.arxiv_id}")
                    if result:
                        return _update_paper(paper, result, "arxiv_id")

            # 3. Try URL as fallback for IDs
            if paper.url:
                result = _get_by_id(client, paper.url)
                if result:
                    return _update_paper(paper, result, "url")

            # 4. Fallback to Title Search
            if paper.title:
                result = _search_by_title(client, paper.title)
                if result:
                    return _update_paper(paper, result, "title_search")
    except Exception as e:
        log.error("semantic_scholar.error", title=paper.title, error=str(e))
        # Ensure we don't crash the agent
        return paper

    # If all failed, ensure citation_count remains None (as per Sprint 1 requirements)
    if paper.citation_count is None:
        log.warning(
            "semantic_scholar.not_found", title=paper.title, arxiv_id=paper.arxiv_id, doi=paper.doi
        )

    return paper


def _get_by_id(client: httpx.Client, paper_id: str) -> dict[str, Any] | None:
    """Retrieve paper details by a specific ID (DOI, ARXIV ID, etc.)."""
    try:
        url = f"{BASE_URL}/{paper_id}"
        params = {"fields": "paperId,citationCount,title,doi"}
        response = client.get(url, params=params)

        if response.status_code == 200:
            return cast(dict[str, Any], response.json())
        elif response.status_code == 429:
            log.warning("semantic_scholar.rate_limit_hit", paper_id=paper_id)
            return None
        elif response.status_code == 404:
            return None
        else:
            log.debug(
                "semantic_scholar.api_status", status_code=response.status_code, paper_id=paper_id
            )
            return None
    except Exception as e:
        log.debug("semantic_scholar.id_lookup_failed", paper_id=paper_id, error=str(e))
        return None


def _search_by_title(client: httpx.Client, title: str) -> dict[str, Any] | None:
    """Search for a paper by title and return the best match."""
    try:
        url = f"{BASE_URL}/search/match"
        params = {"query": title, "fields": "paperId,citationCount,title,doi"}
        response = client.get(url, params=params)

        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                return cast(dict[str, Any], data[0])
        elif response.status_code == 429:
            log.warning("semantic_scholar.rate_limit_hit", title=title)
            return None
        return None
    except Exception as e:
        log.debug("semantic_scholar.title_search_failed", title=title, error=str(e))
        return None


def _update_paper(paper: Paper, result: dict[str, Any], method: str) -> Paper:
    """Update paper fields from Semantic Scholar result."""
    paper.citation_count = result.get("citationCount")
    paper.semantic_scholar_id = result.get("paperId")

    # If DOI was missing but found in Semantic Scholar, populate it
    if not paper.doi and result.get("doi"):
        paper.doi = result.get("doi")

    log.info(
        "semantic_scholar.match_found",
        title=paper.title,
        method=method,
        citations=paper.citation_count,
        ss_id=paper.semantic_scholar_id,
    )
    return paper

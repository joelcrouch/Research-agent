from datetime import UTC, datetime

import arxiv
import structlog

from schemas.paper import Paper

log = structlog.get_logger(__name__)


def parse_date(date_str: str | None) -> datetime | None:
    """
    Parse date string in YYYY-MM-DD format to aware datetime.
    Returns None if parsing fails, allowing the caller to decide
    whether to ignore the filter or fail.
    """
    if not date_str:
        return None

    formats = ["%Y-%m-%d", "%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt).replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue

    log.warning("arxiv.search.invalid_date_format", provided=date_str, expected="YYYY-MM-DD")
    return None


def search_arxiv(
    query: str, max_results: int = 10, date_from: str | None = None, date_to: str | None = None
) -> list[Paper]:
    """
    Search ArXiv for papers and return a list of Paper dataclasses.
    """
    # LLMs sometimes pass a list of terms instead of a single string
    if isinstance(query, list):
        query = " ".join(query)

    # 1. Handle Empty Query
    if not query or not query.strip():
        log.warning("arxiv.search.empty_query", msg="Search requested with empty query string.")
        return []

    try:
        # LLMs sometimes pass strings for numeric arguments
        max_res = int(max_results)
        client = arxiv.Client()

        # 2. Parse Dates
        dt_from = parse_date(date_from)
        dt_to = parse_date(date_to)

        search = arxiv.Search(
            query=query,
            max_results=max_res * 5,  # Fetch more to account for date filtering
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[Paper] = []

        log.info(
            "arxiv.search.start",
            query=query,
            max_results=max_res,
            date_from=date_from,
            date_to=date_to,
        )

        results_iter = client.results(search)
        found_any = False

        for result in results_iter:
            found_any = True
            # Date filtering
            published_date = result.published
            if dt_from and published_date < dt_from:
                continue
            if dt_to and published_date > dt_to:
                continue

            paper = Paper(
                title=result.title,
                author=[author.name for author in result.authors],
                abstract=result.summary,
                year=result.published.year,
                arxiv_id=result.get_short_id().split("v")[0],
                pubmed_id=None,
                doi=result.doi,
                semantic_scholar_id=None,
                citation_count=None,
                source="arxiv",
                url=result.entry_id,
            )
            papers.append(paper)

            if len(papers) >= max_res:
                break
        if not found_any:
            log.info(
                "arxiv.search.no_results",
                query=query,
                msg="API returned zero results for this query.",
            )
        elif not papers and found_any:
            log.info(
                "arxiv.search.filtered_to_zero",
                query=query,
                msg="Results were found but all were filtered out by date range.",
            )

        log.info("arxiv.search.complete", count=len(papers))
        return papers

    except Exception as e:
        log.error("arxiv.search.error", query=query, error=str(e))
        return []

import structlog
import re
from typing import List, Dict, Optional
from schemas.paper import Paper

log = structlog.get_logger(__name__)

def deduplicate(papers: List[Paper]) -> List[Paper]:
    """
    Deduplicate a list of papers based on DOI, ArXiv ID, and normalized title.
    Priority: DOI > ArXiv ID > Normalized Title.
    Merges records to keep the most complete data.
    """
    if not papers:
        return []

    # We use a list to maintain order of first appearance (stability)
    unique_papers: List[Paper] = []
    
    # Indexes for fast lookup
    doi_map: Dict[str, int] = {}      # DOI -> index in unique_papers
    arxiv_map: Dict[str, int] = {}    # ArXiv ID -> index in unique_papers
    title_map: Dict[str, int] = {}    # Normalized Title -> index in unique_papers

    duplicates_found = 0

    for paper in papers:
        match_index = -1
        match_reason = ""

        # 1. Check DOI
        if paper.doi and paper.doi in doi_map:
            match_index = doi_map[paper.doi]
            match_reason = "DOI"
        
        # 2. Check ArXiv ID
        elif paper.arxiv_id and paper.arxiv_id in arxiv_map:
            match_index = arxiv_map[paper.arxiv_id]
            match_reason = "ArXiv ID"

        # 3. Check Title (Fuzzy)
        else:
            norm_title = _normalize_title(paper.title)
            if norm_title and norm_title in title_map:
                match_index = title_map[norm_title]
                match_reason = "Title"

        if match_index != -1:
            # Duplicate found: Merge into the existing record
            existing_paper = unique_papers[match_index]
            _merge_papers(existing_paper, paper)
            duplicates_found += 1
            log.debug("deduplicator.merge", 
                      reason=match_reason, 
                      kept_title=existing_paper.title, 
                      merged_title=paper.title)
            
            # Update maps with new fields from the merged paper (if any)
            if existing_paper.doi:
                doi_map[existing_paper.doi] = match_index
            if existing_paper.arxiv_id:
                arxiv_map[existing_paper.arxiv_id] = match_index
        else:
            # New unique paper
            new_index = len(unique_papers)
            unique_papers.append(paper)
            
            if paper.doi:
                doi_map[paper.doi] = new_index
            if paper.arxiv_id:
                arxiv_map[paper.arxiv_id] = new_index
            
            norm_title = _normalize_title(paper.title)
            if norm_title:
                title_map[norm_title] = new_index

    log.info("deduplicator.complete", input_count=len(papers), output_count=len(unique_papers), duplicates=duplicates_found)
    return unique_papers

def _normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching: lowercase, alphanumeric only."""
    if not title:
        return ""
    # Remove non-alphanumeric characters and convert to lower case
    return re.sub(r'[^a-z0-9]', '', title.lower())

def _merge_papers(target: Paper, source: Paper) -> None:
    """
    Merge source paper into target paper in-place.
    Target keeps its own values unless they are None/Empty and Source has them.
    Specific logic:
    - Lists (authors): Union if possible, or prefer longer list.
    - IDs: Preserve all unique IDs.
    - Citations: Take the non-None one, or the max if both exist.
    """
    # Simple fields: fill if missing
    if not target.abstract and source.abstract:
        target.abstract = source.abstract
    if not target.year and source.year:
        target.year = source.year
    if not target.url and source.url:
        target.url = source.url
    
    # IDs: Ensure we capture all available IDs
    if not target.doi and source.doi:
        target.doi = source.doi
    if not target.arxiv_id and source.arxiv_id:
        target.arxiv_id = source.arxiv_id
    if not target.pubmed_id and source.pubmed_id:
        target.pubmed_id = source.pubmed_id
    if not target.semantic_scholar_id and source.semantic_scholar_id:
        target.semantic_scholar_id = source.semantic_scholar_id
        
    # Citations: Trust the one that exists, or the higher one
    if target.citation_count is None and source.citation_count is not None:
        target.citation_count = source.citation_count
    elif target.citation_count is not None and source.citation_count is not None:
        target.citation_count = max(target.citation_count, source.citation_count)
        
    # Authors: Rough heuristic - prefer the one with more authors or longer strings
    # (Assuming truncation vs full list)
    if not target.author and source.author:
        target.author = source.author
    elif target.author and source.author:
        if len(source.author) > len(target.author):
            target.author = source.author

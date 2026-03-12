from typing import List
from schemas.paper import Paper

def rank_papers(papers: List[Paper], top_k: int) -> List[Paper]:
    """
    Rank papers by citation count descending.
    Papers with citation_count=None are treated as 0 (or simply pushed to bottom).
    Returns top_k results.
    """
    if not papers:
        return []
        
    # Python sort is stable.
    # We want None to be at the bottom.
    # Key function: (citation_count is not None, citation_count)
    # If citation_count is None: (False, None)
    # If citation_count is 10: (True, 10)
    # Sort descending: (True, 10) > (False, None)
    # But wait, (True, 10) vs (True, 5). 10 > 5. Correct.
    
    def sort_key(p: Paper):
        count = p.citation_count
        if count is None:
            return (False, 0) # Treat None as effectively negative/bottom (False < True)
        return (True, count) # Treat real numbers as higher priority
        
    sorted_papers = sorted(
        papers, 
        key=sort_key, 
        reverse=True
    )
    
    return sorted_papers[:top_k]

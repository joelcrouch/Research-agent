from dataclasses import dataclass

@dataclass
class Paper: 
    title: str
    author: list[str]
    abstract: str | None
    year: int | None
    arxiv_id: str | None
    pubmed_id: str | None
    doi: str | None
    semantic_scholar_id: str | None
    citation_count: int | None
    source: str  #arxiv, pubmed semantic_scholar
    url: str
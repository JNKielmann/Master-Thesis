from datetime import datetime
from typing import NamedTuple, List, Dict, Any


class Author(NamedTuple):
    id: str
    name: str
    affiliation: str
    order_in_paper: int


class Keyword(NamedTuple):
    id: str
    value: str


class CitationContext(NamedTuple):
    cited_paper_id: str
    citation_snippets: List[str]


class Paper(NamedTuple):
    id: str
    normalized_title: str
    display_title: str
    publication_date: datetime
    citation_count: int
    total_author_count: int
    authors: List[Author]
    keywords: List[Keyword]
    journal_conference_name: str
    referenced_paper_ids: List[str]
    citation_contexts: List[CitationContext]
    abstract: str

    def to_json(self) -> Dict[str, Any]:
        paper_dict = self._asdict()
        paper_dict["authors"] = [author._asdict() for author in
                                 paper_dict["authors"]]
        paper_dict["keywords"] = [keyword._asdict() for keyword in
                                  paper_dict["keywords"]]
        paper_dict["citation_contexts"] = [cc._asdict() for cc in
                                           paper_dict["citation_contexts"]]
        return paper_dict

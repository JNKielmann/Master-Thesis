from datetime import datetime
from typing import NamedTuple, List, Dict, Any


class Author(NamedTuple):
    id: str
    name: str
    order_in_paper: int


class Keyword(NamedTuple):
    id: str
    value: str


class Journal(NamedTuple):
    id: str
    name: str


class Conference(NamedTuple):
    id: str
    name: str


class Paper(NamedTuple):
    id: str
    normalized_title: str
    display_title: str
    publication_date: datetime
    citation_count: int
    total_author_count: int
    kit_authors: List[Author]
    keywords: List[Keyword]
    journal: Journal
    conference: Conference
    abstract: str

    def to_json(self) -> Dict[str, Any]:
        paper_dict = self._asdict()
        paper_dict["kit_authors"] = [author._asdict() for author in
                                     paper_dict["kit_authors"]]
        paper_dict["keywords"] = [keyword._asdict() for keyword in
                                  paper_dict["keywords"]]
        if paper_dict["journal"] is not None:
            paper_dict["journal"] = paper_dict["journal"]._asdict()
        if paper_dict["conference"] is not None:
            paper_dict["conference"] = paper_dict["conference"]._asdict()
        return paper_dict
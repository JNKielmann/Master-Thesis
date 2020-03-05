import abc
from typing import List

from expert_voting.scored_document import ScoredDocument
from expert_voting.scored_expert import ScoredExpert


class DocumentScoreAggregation(abc.ABC):
    def __init__(self, doc_info):
        self.doc_info = doc_info

    def modify(self, input_list: List[ScoredDocument]) -> List[ScoredExpert]:
        experts = {}
        for doc in input_list:
            authors = self.doc_info[str(doc.id)]["authors"]
            for author in authors:
                if not author["id"] in experts:
                    experts[author["id"]] = ScoredExpert(author["id"], 0)
                expert = experts[author["id"]]
                experts[author["id"]] = expert.change_score(
                    expert.score + self._calc_doc_score(author, len(authors), doc))
        return experts.values()

    @abc.abstractmethod
    def _calc_doc_score(self, author, num_authors, doc):
        pass

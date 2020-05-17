import abc
from typing import List

from expert_voting.scored_document import ScoredDocument
from expert_voting.scored_expert import ScoredExpert


class DocumentScoreAggregation(abc.ABC):
    def __init__(self, doc_info, num_docs=None):
        self.doc_info = doc_info
        self.num_docs = num_docs

    def modify(self, input_list: List[ScoredDocument]) -> List[ScoredExpert]:
        experts = {}
        for doc in input_list:
            authors = self.doc_info[str(doc.id)]["authors"]
            seen_author_ids = set()
            for author in authors:
                if author["id"] in seen_author_ids:
                    continue
                seen_author_ids.add(author["id"])
                if not author["id"] in experts:
                    experts[author["id"]] = {
                        "scores": [],
                        "relevant_docs": [],
                    }
                expert = experts[author["id"]]
                expert["relevant_docs"].append((doc.id, round(doc.score, 3)))
                expert["scores"].append(self._calc_doc_score(author, len(authors), doc))
        return [ScoredExpert(expert_id,
#                              sum(sorted(expert["scores"], reverse=True)[:self.num_docs]),
                             sum(sorted(expert["scores"], reverse=True)[:self.num_docs]) if len(expert["scores"])>1 else 0,
                             expert["relevant_docs"])
                for expert_id, expert in experts.items()]

    @abc.abstractmethod
    def _calc_doc_score(self, author, num_authors, doc):
        pass

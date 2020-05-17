from typing import NamedTuple, List

from expert_voting.scored_document import ScoredDocument


class ScoredExpert(NamedTuple):
    id: int
    score: float
    relevant_docs: List[ScoredDocument]

    def change_score(self, score: float):
        return ScoredExpert(self.id, score, self.relevant_docs)

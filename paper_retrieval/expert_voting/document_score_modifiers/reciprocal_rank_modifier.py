from typing import List

from expert_voting.document_score_modifiers.document_score_modifier import \
    DocumentScoreModifier
from expert_voting.scored_document import ScoredDocument


class ReciprocalRankModifier(DocumentScoreModifier):
    def __init__(self, multiply_score):
        self.multiply_score = multiply_score

    def modify(self, input_list: List[ScoredDocument]) -> List[ScoredDocument]:
        result = []
        for rank, doc in enumerate(input_list):
            new_score = (1 / (rank + 1))
            if self.multiply_score:
                new_score *= doc.score
            result.append(doc.change_score(new_score))
        return result

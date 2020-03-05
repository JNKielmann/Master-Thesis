from typing import List

import numpy as np

from expert_voting.document_score_modifiers.document_score_modifier import \
    DocumentScoreModifier
from expert_voting.scored_document import ScoredDocument


class CitationCountModifier(DocumentScoreModifier):
    def __init__(self, doc_info, alpha):
        self.doc_infos = doc_info
        self.alpha = alpha

    def modify(self, input_list: List[ScoredDocument]) -> List[ScoredDocument]:
        result = []
        for rank, doc in enumerate(input_list):
            citation_count = self.doc_infos[str(doc.id)]["citation_count"]
            new_score = doc.score * np.ln(np.e + citation_count / self.alpha)
            result.append(doc.change_score(new_score))
        return result

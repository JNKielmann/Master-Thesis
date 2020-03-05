import time
from datetime import datetime
from typing import List

import numpy as np

from expert_voting.document_score_modifiers.document_score_modifier import \
    DocumentScoreModifier
from expert_voting.scored_document import ScoredDocument


class PaperAgeModifier(DocumentScoreModifier):
    def __init__(self, doc_info, alpha, beta):
        self.doc_info = doc_info
        self.alpha = alpha
        self.beta = beta

    def modify(self, input_list: List[ScoredDocument]) -> List[ScoredDocument]:
        result = []
        for rank, doc in enumerate(input_list):
            publication_timestamp = self.doc_info[str(doc.id)]["publication_date"]
            publication_timestamp /= (1000*60*60*24)
            current_timestamp = time.time() / (60*60*24)
            paper_age = current_timestamp - publication_timestamp
            new_score = doc.score * ((1 - self.beta)
                                     * (1 / np.power(self.alpha, paper_age))
                                     + self.beta)
            result.append(doc.change_score(new_score))
        return result


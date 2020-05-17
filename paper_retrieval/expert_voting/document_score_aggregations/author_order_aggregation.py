from typing import List

import numpy as np

from expert_voting.document_score_aggregations.document_score_aggregation import \
    DocumentScoreAggregation
from expert_voting.scored_document import ScoredDocument
from expert_voting.scored_expert import ScoredExpert


class AuthorOrderAggregation(DocumentScoreAggregation):
    def __init__(self, doc_info, alpha, num_docs=None):
        super().__init__(doc_info, num_docs)
        self.alpha = alpha

    def _calc_doc_score(self, author, num_authors, doc):
        normalization = np.power(self.alpha, np.arange(num_authors)).sum()
        factor = np.power(self.alpha, author["order_in_paper"]-1) / normalization
        return doc.score * factor

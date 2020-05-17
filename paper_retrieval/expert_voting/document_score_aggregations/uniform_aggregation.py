from typing import List

from expert_voting.document_score_aggregations.document_score_aggregation import \
    DocumentScoreAggregation
from expert_voting.scored_document import ScoredDocument
from expert_voting.scored_expert import ScoredExpert


class UniformAggregation(DocumentScoreAggregation):
    def __init__(self, doc_info, num_docs=None):
        super().__init__(doc_info, num_docs)

    def _calc_doc_score(self, author, num_authors, doc):
        return doc.score/num_authors

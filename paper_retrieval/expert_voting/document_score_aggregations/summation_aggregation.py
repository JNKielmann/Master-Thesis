
from expert_voting.document_score_aggregations.document_score_aggregation import \
    DocumentScoreAggregation


class SummationAggregation(DocumentScoreAggregation):
    def __init__(self, doc_info):
        super().__init__(doc_info)

    def _calc_doc_score(self, author, num_authors, doc):
        return doc.score

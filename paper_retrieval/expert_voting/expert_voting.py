from typing import List

import pandas as pd

from expert_voting.author_score_modifiers.author_score_modifier import AuthorScoreModifier
from expert_voting.document_score_aggregations.document_score_aggregation import \
    DocumentScoreAggregation
from expert_voting.document_score_modifiers.document_score_modifier import \
    DocumentScoreModifier
from expert_voting.scored_document import ScoredDocument
from retrieval_algorithms import RetrievalAlgorithm


class ExpertVoting:
    def __init__(self,
                 doc_score_mods: List[DocumentScoreModifier],
                 doc_score_agg: DocumentScoreAggregation,
                 author_score_mods: List[AuthorScoreModifier],
                 document_retrieval: RetrievalAlgorithm):
        self.doc_score_mods = doc_score_mods
        self.doc_score_agg = doc_score_agg
        self.author_score_mods = author_score_mods
        self.document_retrieval = document_retrieval

    def get_ranking(self, query, author_info=None) -> pd.DataFrame:
        document_ranking = self.document_retrieval.get_ranking(query)
        scored_documents = [ScoredDocument(doc[1], doc[2])
                            for doc in document_ranking.to_records()]
        modified_doc_score = scored_documents
        for doc_score_mod in self.doc_score_mods:
            modified_doc_score = doc_score_mod.modify(modified_doc_score)
        author_score = self.doc_score_agg.modify(modified_doc_score)
        modified_author_score = author_score
        for author_score_mod in self.author_score_mods:
            modified_author_score = author_score_mod.modify(modified_author_score)
        result = pd.DataFrame(modified_author_score)
        result["#rel docs"] = result["relevant_docs"].apply(len)
        result.sort_values(by="score", ascending=False, inplace=True)
        result["score"] = result["score"].round(3)
        if author_info is not None:
            result["name"] = result["id"].map(
                lambda author_id: author_info[str(author_id)]["name"])
        return result

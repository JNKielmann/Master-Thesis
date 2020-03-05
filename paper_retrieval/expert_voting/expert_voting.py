from typing import List

from expert_voting.author_score_modifiers.author_score_modifier import AuthorScoreModifier
from expert_voting.document_score_aggregations.document_score_aggregation import \
    DocumentScoreAggregation
from expert_voting.document_score_modifiers.document_score_modifier import \
    DocumentScoreModifier
from expert_voting.scored_document import ScoredDocument
from expert_voting.scored_expert import ScoredExpert


class ExpertVoting:
    def __init__(self,
                 doc_score_mods: List[DocumentScoreModifier],
                 doc_score_agg: DocumentScoreAggregation,
                 author_score_mods: List[AuthorScoreModifier]):
        self.doc_score_mods = doc_score_mods
        self.doc_score_agg = doc_score_agg
        self.author_score_mods = author_score_mods

    def vote(self, input_list: List[ScoredDocument]) -> List[ScoredExpert]:
        modified_doc_score = input_list
        for doc_score_mod in self.doc_score_mods:
            modified_doc_score = doc_score_mod.modify(modified_doc_score)
        author_score = self.doc_score_agg.modify(modified_doc_score)
        modified_author_score = author_score
        for author_score_mod in self.author_score_mods:
            modified_author_score = author_score_mod.modify(modified_author_score)
        return sorted(modified_author_score, key=lambda x: x.score, reverse=True)

from typing import List

import numpy as np

from expert_voting.author_score_modifiers.author_score_modifier import AuthorScoreModifier
from expert_voting.scored_expert import ScoredExpert


class PaperCountModifier(AuthorScoreModifier):
    def __init__(self, doc_info):
        self.author_paper_counts = {}
        for paper in doc_info.values():
            for author in paper["authors"]:
                if str(author["id"]) not in self.author_paper_counts:
                    self.author_paper_counts[str(author["id"])] = 0
                self.author_paper_counts[str(author["id"])] += 1

    def modify(self, input_list: List[ScoredExpert]) -> List[ScoredExpert]:
        result = []
        for author in input_list:
            new_score = author.score / np.log(1+self.author_paper_counts[str(author.id)])
            result.append(author.change_score(new_score))
        return result

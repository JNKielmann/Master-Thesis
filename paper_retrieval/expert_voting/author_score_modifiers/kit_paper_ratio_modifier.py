from typing import List

from expert_voting.author_score_modifiers.author_score_modifier import AuthorScoreModifier
from expert_voting.scored_expert import ScoredExpert


class KitPaperRatioModifier(AuthorScoreModifier):
    def __init__(self, author_info, alpha):
        self.author_info = author_info
        self.alpha = alpha

    def modify(self, input_list: List[ScoredExpert]) -> List[ScoredExpert]:
        result = []
        for author in input_list:
            kit_paper = self.author_info("kit_paper") + self.alpha
            total_paper = (self.author_info["kit_paper"] +
                           self.author_info["non_kit_paper"] +
                           self.alpha)
            new_score = author.score * (kit_paper / total_paper)
            result.append(author.change_score(new_score))
        return result

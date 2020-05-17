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
            info = self.author_info[str(author.id)]
            kit_paper = info["kit_paper"] + self.alpha
            total_paper = info["kit_paper"] + info["non_kit_paper"] + self.alpha
            if total_paper == 0:
                total_paper = 1
            new_score = author.score * (kit_paper / total_paper)
#             if (kit_paper / total_paper) < 0.2:
#                 new_score = 0
            result.append(author.change_score(new_score))
        return result

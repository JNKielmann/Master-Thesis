import time
from typing import List
import numpy as np

from expert_voting.author_score_modifiers.author_score_modifier import AuthorScoreModifier
from expert_voting.scored_expert import ScoredExpert


class LastKitPaperModifier(AuthorScoreModifier):
    def __init__(self, author_info, alpha, beta):
        self.author_info = author_info
        self.alpha = alpha
        self.beta = beta

    def modify(self, input_list: List[ScoredExpert]) -> List[ScoredExpert]:
        result = []
        for author in input_list:
            last_publication_timestamp = self.author_info[str(author.id)][
                "last_publication_date"]
            last_publication_timestamp /= (1000 * 60 * 60 * 24 * 365)
            current_timestamp = time.time() / (60 * 60 * 24 * 365)
            age = current_timestamp - last_publication_timestamp
            new_score = author.score * ((1 - self.beta)
                                        * (1 / np.power(self.alpha, age))
                                        + self.beta)
            result.append(author.change_score(new_score))
        return result

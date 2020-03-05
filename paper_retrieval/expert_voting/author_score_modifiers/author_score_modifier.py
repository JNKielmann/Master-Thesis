import abc
from typing import List

from expert_voting.scored_expert import ScoredExpert


class AuthorScoreModifier(abc.ABC):
    @abc.abstractmethod
    def modify(self, input_list: List[ScoredExpert]) -> List[ScoredExpert]:
        pass

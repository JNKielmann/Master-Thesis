import abc
from typing import List

from expert_voting.scored_document import ScoredDocument


class DocumentScoreModifier(abc.ABC):
    @abc.abstractmethod
    def modify(self, input_list: List[ScoredDocument]) -> List[ScoredDocument]:
        pass

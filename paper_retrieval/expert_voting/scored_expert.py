from typing import NamedTuple


class ScoredExpert(NamedTuple):
    id: int
    score: float

    def change_score(self, score: float):
        return ScoredExpert(self.id, score)

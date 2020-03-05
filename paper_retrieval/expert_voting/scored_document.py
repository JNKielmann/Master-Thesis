from typing import NamedTuple


class ScoredDocument(NamedTuple):
    id: int
    score: float

    def change_score(self, score: float):
        return ScoredDocument(self.id, score)

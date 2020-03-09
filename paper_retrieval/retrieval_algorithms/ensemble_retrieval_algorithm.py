import pandas as pd

from preprocessing import Corpus
from retrieval_algorithms import RetrievalAlgorithm


class EnsembleRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, algo1: RetrievalAlgorithm, algo2: RetrievalAlgorithm, weight):
        self.algo1 = algo1
        self.algo2 = algo2
        self.weight = weight

    def prepare(self, corpus: Corpus):
        self.algo1.prepare(corpus)
        self.algo2.prepare(corpus)

    def get_ranking(self, query: str) -> pd.DataFrame:
        ranking1 = self.algo1.get_ranking(query)
        ranking2 = self.algo2.get_ranking(query)
        df = pd.merge(ranking1, ranking2, how="outer", on="id")
        df = df.fillna(0)
        df["score"] = (self.weight * df["score_x"] +
                       (1 - self.weight) * df["score_y"])
        df.sort_values(by="score", ascending=False, inplace=True)
        return df[["id", "score"]]

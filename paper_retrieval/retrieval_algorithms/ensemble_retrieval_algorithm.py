import pickle

import pandas as pd

from preprocessing import Corpus
from retrieval_algorithms.retrieval_algorithm import RetrievalAlgorithm


class EnsembleRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, model1_file: RetrievalAlgorithm, model2_file: RetrievalAlgorithm,
                 weight: float):
        self.model1_file = model1_file
        self.model2_file = model2_file
        self.weight = weight
        self.model1 = None
        self.model2 = None

    def prepare(self, corpus: Corpus):
        if self.model1 is not None:
            return
        with open(self.model1_file, "rb") as file:
            self.model1 = pickle.load(file)
        with open(self.model2_file, "rb") as file:
            self.model2 = pickle.load(file)

    def get_ranking(self, query: str) -> pd.DataFrame:
        ranking1 = self.model1.get_ranking(query)
        ranking2 = self.model2.get_ranking(query)
        df = pd.merge(ranking1, ranking2, how="outer", on="id")
        df = df.fillna(0)
        df["score"] = (self.weight * df["score_x"] +
                       (1 - self.weight) * df["score_y"])
        df.sort_values(by="score", ascending=False, inplace=True)
        return df[["id", "score"]]

    def __getstate__(self):
        return self.model1_file, self.model2_file, self.weight

    def __setstate__(self, state):
        self.model1_file, self.model2_file, self.weight = state
        self.model1 = self.model2 = None
        self.prepare(None)

import pickle
from typing import List
import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from gensim.summarization.bm25 import BM25


class BM25Retrieval:
    def __init__(self,
                 ids: List[str],
                 corpus: List[List[str]],
                 use_bigrams: bool):
        self.ids = pd.Series(ids, name="id")
        self.phraser = None
        if use_bigrams:
            phrases = Phrases(corpus, min_count=2, threshold=15)
            corpus = phrases[corpus]
            self.phraser = Phraser(phrases)
        self.bm25_model = BM25(corpus)

    def get_ranked_documents(self, query: List[str]) -> pd.DataFrame:
        if self.phraser:
            query = self.phraser[query]
        df = pd.DataFrame(self.ids)
        df["score"] = self.bm25_model.get_scores(query)
        df.sort_values(by="score", ascending=False, inplace=True)
        return df[df["score"] > 0]

    def save(self, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> "BM25Retrieval":
        with open(file_path, "rb") as file:
            return pickle.load(file)

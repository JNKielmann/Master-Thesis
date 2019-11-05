import pickle
from typing import List
import pandas as pd
from gensim.summarization.bm25 import BM25

from preprocessing import apply_pipeline, Corpus


class BM25Retrieval:
    def __init__(self, corpus: Corpus):
        self.ids = pd.Series(corpus.ids, name="id")
        self.bm25_model = BM25(corpus.data)
        self.pipeline = corpus.pipeline

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        # print(query)
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

import pickle
import pandas as pd
from gensim.summarization import bm25

from preprocessing import apply_pipeline, Corpus


class BM25Retrieval:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        self.ids = pd.Series(corpus.ids, name="id")
        self.bm25_model = bm25.BM25(corpus.data)
        self.pipeline = corpus.pipeline
        self.k1 = k1
        self.b = b

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        df = pd.DataFrame(self.ids)
        bm25.PARAM_K1 = self.k1
        bm25.PARAM_B = self.b
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

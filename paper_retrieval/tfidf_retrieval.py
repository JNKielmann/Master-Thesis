import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

from bm25_transformer import BM25Transformer
from preprocessing import Corpus, apply_pipeline


class TfidfRetrieval:
    def __init__(self, corpus: Corpus, use_idf=True, sublinear_tf=True,
                 max_ngram=1, use_bm25=False, k1=1.0, b=0.75):
        self.pipeline = corpus.pipeline
        self.ids = pd.Series(corpus.ids, name="id")
        self.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, max_ngram),
            min_df=2
        )
        if use_bm25:
            self.tfidf_transformer = BM25Transformer(use_idf, k1, b)
        else:
            self.tfidf_transformer = TfidfTransformer(
                use_idf=use_idf,
                sublinear_tf=sublinear_tf
            )
        term_freq = self.vectorizer.fit_transform(corpus.data)
        self.vectorized_corpus = self.tfidf_transformer.fit_transform(term_freq)

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        vectorized_query = self.vectorizer.transform([query])
        vectorized_query = self.tfidf_transformer.transform(vectorized_query)
        df = pd.DataFrame(self.ids)
        df["score"] = linear_kernel(vectorized_query, self.vectorized_corpus).flatten()
        df.sort_values(by="score", ascending=False, inplace=True)
        return df[df["score"] > 0]

    def save(self, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> "TfidfRetrieval":
        with open(file_path, "rb") as file:
            return pickle.load(file)


def identity(x):
    return x

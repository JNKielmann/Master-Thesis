import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from bm25_transformer import BM25Transformer
from preprocessing import Corpus, apply_pipeline
from retrieval_algorithms import RetrievalAlgorithm
from .identity import identity


class TfIdfRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, k1, b, max_ngram=1, min_df=1):
        self.k1 = k1
        self.b = b
        self.max_ngram = max_ngram
        self.min_df = min_df
        self.pipeline = None
        self.ids = None
        self.count_vectorizer = None
        self.vectorized_corpus = None

    def prepare(self, corpus: Corpus):
        self.pipeline = corpus.pipeline
        self.ids = pd.Series(corpus.ids, name="id")
        self.count_vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, self.max_ngram),
            min_df=self.min_df
        )
        bm25_vectorizer = BM25Transformer(
            k1=self.k1,
            b=self.b
        )
        self.vectorized_corpus = self.count_vectorizer.fit_transform(corpus.data)
        self.vectorized_corpus = bm25_vectorizer.fit_transform(self.vectorized_corpus)

    def get_ranking(self, query: str) -> pd.DataFrame:
        if self.ids is None:
            raise RuntimeError("Prepare of class TfIdfRetrievalAlgorithm has to be "
                               "called before using get_ranking")
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        vectorized_query = self.count_vectorizer.transform([query])

        df = pd.DataFrame(self.ids)
        df["score"] = (self.vectorized_corpus * vectorized_query.T).toarray()
        df.sort_values(by="score", ascending=False, inplace=True)
        return df[df["score"] > 0]

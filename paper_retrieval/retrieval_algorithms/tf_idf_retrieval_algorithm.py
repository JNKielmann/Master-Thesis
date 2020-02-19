import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from preprocessing import Corpus, apply_pipeline
from retrieval_algorithms import RetrievalAlgorithm, identity


class TfIdfRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, use_idf=True, sublinear_tf=True, max_ngram=1, min_df=1):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.max_ngram = max_ngram
        self.min_df = min_df
        self.pipeline = None
        self.ids = None
        self.vectorizer = None
        self.vectorized_corpus = None

    def prepare(self, corpus: Corpus):
        self.pipeline = corpus.pipeline
        self.ids = pd.Series(corpus.ids, name="id")
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, self.max_ngram),
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df
        )
        self.vectorized_corpus = self.vectorizer.fit_transform(corpus.data)
        self.vectorized_corpus = normalize(self.vectorized_corpus)

    def get_ranking(self, query: str) -> pd.DataFrame:
        if self.ids is None:
            raise RuntimeError("Prepare of class TfIdfRetrievalAlgorithm has to be "
                               "called before using get_ranking")
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        vectorized_query = self.vectorizer.transform([query])
        vectorized_query = normalize(vectorized_query)

        df = pd.DataFrame(self.ids)
        df["score"] = (self.vectorized_corpus * vectorized_query.T).toarray()
        df.sort_values(by="score", ascending=False, inplace=True)
        return df[df["score"] > 0]

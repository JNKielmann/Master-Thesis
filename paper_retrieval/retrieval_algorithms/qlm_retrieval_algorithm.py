import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import Corpus, apply_pipeline
from .retrieval_algorithm import RetrievalAlgorithm
from .identity import identity


class QueryLMRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, smoothing_method, smoothing_weight, max_ngram=1, min_df=2):
        self.pipeline = None
        self.ids = None
        self.smoothing_method = smoothing_method
        self.smoothing_weight = smoothing_weight
        self.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, max_ngram),
            min_df=min_df
        )
        self.document_term_probs = None
        self.corpus_model = None
        self.document_lengths = None

    def prepare(self, corpus: Corpus):
        self.pipeline = corpus.pipeline
        self.ids = pd.Series(corpus.ids, name="id")
        document_term_counts = self.vectorizer.fit_transform(corpus.data)
        # divide by document length to get mle estimation
        self.document_lengths = np.array(document_term_counts.sum(axis=1)).squeeze()
        if self.smoothing_method is None or self.smoothing_method == "jm":
            sparse_dl = self.document_lengths[document_term_counts.nonzero()[0]]
            document_term_counts.data = (document_term_counts.data / sparse_dl)
        self.document_term_probs = document_term_counts

        # compute corpus model for smoothing
        corpus_model = self.vectorizer.transform([np.concatenate(corpus.data)])
        corpus_model = corpus_model.toarray().squeeze()
        corpus_model = corpus_model / corpus_model.sum()
        self.corpus_model = corpus_model

    def get_ranking(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        query_term_counts = self.vectorizer.transform([query])
        used_query_terms = query_term_counts.nonzero()[1]
        used_document_term_probs = self.document_term_probs[:, used_query_terms].toarray()
        corpus_probs = self.corpus_model[used_query_terms]
        if self.smoothing_method == "jm":
            smoothed_probs = (self.smoothing_weight * used_document_term_probs +
                              (1 - self.smoothing_weight) * corpus_probs)
        elif self.smoothing_method == "dp":
            a = (used_document_term_probs + self.smoothing_weight * corpus_probs)
            b = (self.document_lengths + self.smoothing_weight).reshape(-1, 1)
            smoothed_probs = a / b
        elif self.smoothing_method is None:
            smoothed_probs = used_document_term_probs
        else:
            raise RuntimeError(f"Smoothing method {self.smoothing_method} is not known!")
        smoothed_probs **= query_term_counts.toarray()[0, used_query_terms]
        document_scores = np.prod(smoothed_probs, axis=1)

        # log probabilities
        # smoothed_probs = np.log(smoothed_probs)
        # smoothed_probs *= query_term_counts.toarray()[0, used_query_terms]
        # document_scores = np.sum(smoothed_probs, axis=1)

        df = pd.DataFrame(self.ids)
        df["score"] = document_scores
        df.sort_values(by="score", ascending=False, inplace=True)
        return df

import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import Corpus, apply_pipeline


class QLMRetrieval:
    def __init__(self, corpus: Corpus, max_ngram=1):
        self.pipeline = corpus.pipeline
        self.ids = pd.Series(corpus.ids, name="id")
        self.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, max_ngram),
        )
        document_term_counts = self.vectorizer.fit_transform(corpus.data)
        # divide by document length to get mle estimation
        document_lengths = np.array(document_term_counts.sum(axis=1)).squeeze()
        document_term_counts.data = (document_term_counts.data /
                                     document_lengths[document_term_counts.nonzero()[0]])
        self.document_term_counts = document_term_counts

        # compute corpus model for smoothing
        corpus_model = self.vectorizer.transform([np.concatenate(corpus.data)])
        corpus_model = corpus_model.toarray().squeeze()
        corpus_model = corpus_model / corpus_model.sum()
        self.corpus_model = corpus_model

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        query_term_counts = self.vectorizer.transform([query])
        used_query_terms = query_term_counts.nonzero()[1]
        document_probs = self.document_term_counts[:, used_query_terms].toarray()
        corpus_probs = self.corpus_model[used_query_terms]
        smoothed_probs = 0.8 * document_probs + 0.2 * corpus_probs
        smoothed_probs **= query_term_counts.toarray()[0, used_query_terms]
        document_scores = np.prod(smoothed_probs, axis=1)
        # smoothed_probs = np.log(smoothed_probs)
        # smoothed_probs *= query_term_counts.toarray()[0, used_query_terms]
        # document_scores = np.sum(smoothed_probs, axis=1)


        df = pd.DataFrame(self.ids)
        df["score"] = document_scores
        df.sort_values(by="score", ascending=False, inplace=True)
        return df

    def save(self, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> "QLMRetrieval":
        with open(file_path, "rb") as file:
            return pickle.load(file)


def identity(x):
    return x

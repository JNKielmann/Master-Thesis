import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import Corpus
from .identity import identity
from retrieval_algorithms import RetrievalAlgorithm


class PRFWrapper(RetrievalAlgorithm):
    def __init__(self,
                 retrieval_algorithm: RetrievalAlgorithm,
                 num_relevant_docs: int,
                 num_expansion_terms: int):
        self.retrieval_algorithm = retrieval_algorithm
        self.num_relevant_docs = num_relevant_docs
        self.num_expansion_terms = num_expansion_terms
        self.vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, 2),
        )
        self.ids = None
        self.vectorized_corpus = None
        self.id_to_word = None

    def prepare(self, corpus: Corpus):
        self.ids = pd.Series(corpus.ids, name="id")
        self.retrieval_algorithm.prepare(corpus)
        self.vectorized_corpus = self.vectorizer.fit_transform(corpus.data)
        self.id_to_word = np.array(self.vectorizer.get_feature_names())

    def get_ranking(self, query: str) -> pd.DataFrame:
        ranked_documents = self.retrieval_algorithm.get_ranking(query)
        word_counts = self.vectorized_corpus[ranked_documents.index, :].sum(axis=0)
        expansion_terms = self.id_to_word[
            np.argsort(word_counts)[::-1][:self.num_expansion_terms]]

        ranked_documents_expanded = self.retrieval_algorithm.get_ranking(
            " ".join(expansion_terms))
        joined_documents = pd.merge(ranked_documents, ranked_documents_expanded, on="id",
                                    how="outer")
        joined_documents.fillna(0)
        joined_documents["score"] = (0.8 * joined_documents["score_x"] +
                                     0.2 * joined_documents["score_y"])
        return joined_documents[["id", "score"]]

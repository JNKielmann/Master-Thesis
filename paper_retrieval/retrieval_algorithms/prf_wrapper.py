import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import Corpus
from retrieval_algorithms import RetrievalAlgorithm
from .identity import identity


class PRFWrapper(RetrievalAlgorithm):
    def __init__(self,
                 retrieval_algorithm: RetrievalAlgorithm,
                 num_relevant_docs: int,
                 num_expansion_terms: int,
                 expansion_weight: int,
                 max_ngram=2):
        self.retrieval_algorithm = retrieval_algorithm
        self.num_relevant_docs = num_relevant_docs
        self.num_expansion_terms = num_expansion_terms
        self.expansion_weight = expansion_weight
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, max_ngram),
            min_df=2
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
        if len(ranked_documents) == 0:
            return ranked_documents
        word_counts = self.vectorized_corpus[
                      ranked_documents.index[:self.num_relevant_docs], :].sum(
            axis=0).getA1()
        top_words = np.argsort(word_counts)[::-1][:self.num_expansion_terms]
        top_words_count = word_counts[top_words]
        expansion_terms = self.id_to_word[top_words]

        ranked_documents_expanded = self.retrieval_algorithm.get_ranking(
            expansion_terms, top_words_count / top_words_count.sum())
        joined_documents = pd.merge(ranked_documents, ranked_documents_expanded, on="id",
                                    how="outer")
        joined_documents = joined_documents.fillna(0)
        score_x = self.expansion_weight * joined_documents["score_x"]
        score_y = (1 - self.expansion_weight) * joined_documents["score_y"]
        joined_documents["score"] = (score_x + score_y)
        joined_documents.sort_values(by="score", ascending=False, inplace=True)
        return joined_documents[["id", "score"]]

    def get_expansion_terms(self, query):
        ranked_documents = self.retrieval_algorithm.get_ranking(query)
        if len(ranked_documents) == 0:
            return ranked_documents
        word_counts = self.vectorized_corpus[
                      ranked_documents.index[:self.num_relevant_docs], :].sum(
            axis=0).getA1()
        top_words = np.argsort(word_counts)[::-1][:self.num_expansion_terms]
        top_words_count = word_counts[top_words]
        expansion_terms = self.id_to_word[top_words]
        return dict([term for term in zip(expansion_terms, top_words_count) if
                     term[0] not in query.split(" ")])

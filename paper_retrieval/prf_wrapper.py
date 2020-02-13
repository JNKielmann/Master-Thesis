import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import Corpus, apply_pipeline


class PRFWrapper:
    def __init__(self, corpus: Corpus, retrieval_algorithm):
        self.corpus = corpus
        self.ids = pd.Series(corpus.ids, name="id")
        self.retrieval_algorithm = retrieval_algorithm

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        ranked_documents = self.retrieval_algorithm.get_ranked_documents(query)
        relevant_text = self.corpus.data[ranked_documents[:20].index].sum()
        vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            ngram_range=(1, 1),
        )
        word_counts = vectorizer.fit_transform([relevant_text]).toarray().squeeze()
        id_to_word = np.array(vectorizer.get_feature_names())
        expansion_terms = id_to_word[np.argsort(word_counts)[::-1][:20]]

        return self.retrieval_algorithm.get_ranked_documents(" ".join(expansion_terms))


def identity(x):
    return x

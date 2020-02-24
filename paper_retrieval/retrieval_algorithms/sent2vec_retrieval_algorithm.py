import logging

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sent2vec import Sent2vecModel
from sklearn.preprocessing import normalize

from preprocessing import Corpus, apply_pipeline
from retrieval_algorithms import RetrievalAlgorithm


class Sent2VecRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, model_path):
        self.pipeline = None
        self.ids = None
        self.model_path = model_path
        self.sent2vec_model = None
        self.document_lookup = None

    def prepare(self, corpus: Corpus):
        self.ids = corpus.ids
        self.pipeline = corpus.pipeline
        self.sent2vec_model = Sent2vecModel()
        logging.info(f"Started loading sent2vec model from {self.model_path}")
        self.sent2vec_model.load_model(self.model_path)
        logging.info(f"Finished loading sent2vec model from {self.model_path}")

        doc_vectors = corpus.data.str.join(" ")\
            .progress_apply(self.sent2vec_model.embed_sentence)
        self.document_lookup = KeyedVectors(self.sent2vec_model.get_emb_size())
        self.document_lookup.add(corpus.ids, np.stack(doc_vectors.values).squeeze())

    def get_ranking(self, query: str) -> pd.DataFrame:
        if self.ids is None:
            raise RuntimeError("Prepare of class Sent2VecRetrievalAlgorithm has to be "
                               "called before using get_ranking")
        query = apply_pipeline(query, self.pipeline)
        query = self.sent2vec_model.embed_sentence(query).squeeze()
        relevant_docs = self.document_lookup.similar_by_vector(query, topn=100000)
        return pd.DataFrame(relevant_docs, columns=["id", "score"])

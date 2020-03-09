import logging

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sent2vec import Sent2vecModel
from sklearn.preprocessing import normalize

from preprocessing import Corpus, apply_pipeline
from retrieval_algorithms import RetrievalAlgorithm


class Sent2VecRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, model_path, use_annoy):
        self.pipeline = None
        self.ids = None
        self.model_path = model_path
        self.use_annoy = use_annoy
        self.sent2vec_model = None
        self.document_lookup = None
        self.annoy_index = None
        self.doc_vectors = None

    def prepare(self, corpus: Corpus):
        if self.ids is not None:
            return
        self.ids = corpus.ids
        self.pipeline = corpus.pipeline
        self.sent2vec_model = Sent2vecModel()
        logging.info(f"Started loading sent2vec model from {self.model_path}")
        self.sent2vec_model.load_model(self.model_path)
        logging.info(f"Finished loading sent2vec model from {self.model_path}")

        doc_vectors = corpus.data.str.join(" ") \
            .progress_apply(self.sent2vec_model.embed_sentence)

        if self.use_annoy:
            self.annoy_index = AnnoyIndex(self.sent2vec_model.get_emb_size(),
                                          'angular')
            for index, item in doc_vectors.iteritems():
                self.annoy_index.add_item(index, item.squeeze())
            self.annoy_index.build(10)  # 10 trees
        else:
            print(np.stack(doc_vectors.values).squeeze().shape)
            self.doc_vectors = normalize(np.stack(doc_vectors.values).squeeze())

    def get_ranking(self, query: str) -> pd.DataFrame:
        if self.ids is None:
            raise RuntimeError("Prepare of class Sent2VecRetrievalAlgorithm has to be "
                               "called before using get_ranking")
        query = apply_pipeline(query, self.pipeline)
        query = self.sent2vec_model.embed_sentence(query).squeeze()
        # relevant_docs = self.document_lookup.similar_by_vector(query, topn=100000)
        if self.use_annoy:
            relevant_docs = self.annoy_index.get_nns_by_vector(query, 10000,
                                                               include_distances=True)
            return pd.DataFrame({
                "id": self.ids[relevant_docs[0]].values,
                "score": relevant_docs[1]
            })
        else:
            query = normalize(query.reshape(1, -1)).squeeze()
            return pd.DataFrame({
                "id": self.ids,
                "score": self.doc_vectors.dot(query)
            }).sort_values(by="score", ascending=False)

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model, load_facebook_vectors, FastText
from gensim.models.keyedvectors import FastTextKeyedVectors

from preprocessing import Corpus, apply_pipeline


class WordEmbeddingRetrieval:
    def __init__(self, corpus: Corpus, model: FastTextKeyedVectors):
        self.ids = corpus.ids
        self.pipeline = corpus.pipeline
        self.model = model
        self.model.init_sims(True)
        doc_vectors = corpus.data.progress_apply(self._get_average_embedding)
        self.document_lookup = KeyedVectors(self.model.vector_size)
        self.document_lookup.add(corpus.ids, np.stack(doc_vectors.values))

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        query = self._get_average_embedding(query)
        if query is None:
            relevant_docs = []
        else:
            relevant_docs = self.document_lookup.similar_by_vector(query, topn=1000)
        return pd.DataFrame(relevant_docs, columns=["id", "score"])

    def _get_average_embedding(self, text):
        words = [word for word in text if word in self.model.vocab]
        if len(words) == 0:
            return None
        return self.model[words].mean(axis=0)

    @staticmethod
    def from_pretrained_embedding(
            corpus: Corpus, pretrained_model_path: str
    ) -> 'WordEmbeddingRetrieval':
        print("from_pretrained_embedding")
        pretrained_vectors = load_facebook_vectors(pretrained_model_path)
        return WordEmbeddingRetrieval(corpus, pretrained_vectors)

    @staticmethod
    def from_new_embedding(corpus: Corpus, window_size=5) -> 'WordEmbeddingRetrieval':
        print("from_new_embedding")
        model = FastText(sg=True, size=300, min_count=1, window=window_size)
        model.build_vocab(sentences=corpus.data)
        model.train(sentences=corpus.data, total_examples=len(corpus.data), epochs=20)
        return WordEmbeddingRetrieval(corpus, model.wv)

    @staticmethod
    def from_finetuned_embedding(
            corpus: Corpus, pretrained_model_path: str
    ) -> 'WordEmbeddingRetrieval':
        print("from_finetuned_embedding")
        model = load_facebook_model(pretrained_model_path)
        model.min_count = 1
        model.build_vocab(sentences=corpus.data, update=True)
        model.train(sentences=corpus.data, total_examples=len(corpus.data),
                    epochs=20)
        return WordEmbeddingRetrieval(corpus, model.wv)

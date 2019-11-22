from typing import List

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model, load_facebook_vectors, FastText
from gensim.models.keyedvectors import FastTextKeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import Corpus, apply_pipeline


class WordEmbeddingRetrieval:
    def __init__(self, corpus: Corpus, model: FastTextKeyedVectors, sentence_embedder):
        self.ids = corpus.ids
        self.pipeline = corpus.pipeline
        self.model = model
        self.model.init_sims(True)
        self.sentence_embedder = sentence_embedder(self.model, corpus)
        doc_vectors = corpus.data.progress_apply(self.sentence_embedder.encode_sentence)
        self.document_lookup = KeyedVectors(self.model.vector_size)
        self.document_lookup.add(corpus.ids, np.stack(doc_vectors.values))

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        query = self.sentence_embedder.encode_sentence(query)
        if query is None:
            relevant_docs = []
        else:
            relevant_docs = self.document_lookup.similar_by_vector(query, topn=1000)
        return pd.DataFrame(relevant_docs, columns=["id", "score"])

    @staticmethod
    def from_pretrained_embedding(corpus: Corpus, sentence_embedder,
                                  pretrained_model_path: str) -> 'WordEmbeddingRetrieval':
        print("from_pretrained_embedding")
        pretrained_vectors = load_facebook_vectors(pretrained_model_path)
        return WordEmbeddingRetrieval(corpus, pretrained_vectors, sentence_embedder)

    @staticmethod
    def from_new_embedding(corpus: Corpus, sentence_embedder,
                           window_size=5, embedding_size=300) -> 'WordEmbeddingRetrieval':
        print("from_new_embedding")
        model = FastText(sg=True, size=embedding_size, min_count=1, window=window_size,
                         negative=10)
        model.build_vocab(sentences=corpus.data)
        model.train(sentences=corpus.data, total_examples=len(corpus.data), epochs=20)
        return WordEmbeddingRetrieval(corpus, model.wv, sentence_embedder)

    @staticmethod
    def from_finetuned_embedding(corpus: Corpus, sentence_embedder,
                                 pretrained_model_path: str) -> 'WordEmbeddingRetrieval':
        print("from_finetuned_embedding")
        model = load_facebook_model(pretrained_model_path)
        model.min_count = 1
        model.build_vocab(sentences=corpus.data, update=True)
        model.train(sentences=corpus.data, total_examples=len(corpus.data),
                    epochs=20)
        return WordEmbeddingRetrieval(corpus, model.wv, sentence_embedder)


class AverageSentenceEmbedding:
    def __init__(self, model, corpus):
        self.model = model

    def encode_sentence(self, sentence: List[str]):
        words = [word for word in sentence if word in self.model.vocab]
        if len(words) == 0:
            return None
        return self.model[words].mean(axis=0)


class TfidfSentenceEmbedding:
    def __init__(self, model, corpus):
        self.model = model
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=identity,
            preprocessor=identity,
            use_idf=True,
            sublinear_tf=True
        )
        self.vectorizer.fit(corpus.data)

    def encode_sentence(self, sentence: List[str]):
        words = [word for word in sentence if word in self.model.vocab]
        if len(words) == 0:
            return None
        word_ids = [self.vectorizer.vocabulary_[word] for word in words]
        idf = self.vectorizer.idf_[word_ids].reshape(-1, 1)
        return np.sum(self.model[words] * idf, axis=0) / len(words)


class SifSentenceEmbedding:
    def __init__(self, model, corpus):
        self.model = model
        self.vlookup = model.vocab  # Gives us access to word index and count

        self.alpha = 1e-3
        self.Z = 0
        for k in self.vlookup:
            self.Z += self.vlookup[k].count  # Compute the normalization constant Z

    def encode_sentence(self, sentence: List[str]):
        v = np.zeros(self.model.vector_size)
        words = [word for word in sentence if word in self.model.vocab]
        if len(words) == 0:
            return None
        for w in words:
            v += (self.alpha / (self.alpha + (self.vlookup[w].count / self.Z))) * \
                 self.model[w]
        v /= len(words)
        return v


def identity(x):
    return x

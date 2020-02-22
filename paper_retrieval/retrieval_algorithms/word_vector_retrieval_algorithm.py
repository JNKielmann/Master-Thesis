from typing import List

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors, load_facebook_model, FastText
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import Corpus, apply_pipeline
from retrieval_algorithms import RetrievalAlgorithm, identity


class WordVectorRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, embedding, sentence_embedder):
        self.pipeline = None
        self.ids = None
        self.embedding = embedding
        self.sentence_embedder = sentence_embedder
        self.model = None
        self.document_lookup = None

    def prepare(self, corpus: Corpus):
        self.pipeline = corpus.pipeline
        self.ids = pd.Series(corpus.ids, name="id")
        self.model = self.embedding.train_model()
        self.sentence_embedder = self.sentence_embedder(self.model, corpus)
        doc_vectors = corpus.data.progress_apply(self.sentence_embedder.encode_sentence)
        self.document_lookup = KeyedVectors(self.model.vector_size)
        self.document_lookup.add(corpus.ids, np.stack(doc_vectors.values))

    def get_ranking(self, query: str) -> pd.DataFrame:
        if self.ids is None:
            raise RuntimeError("Prepare of class WordVectorRetrievalAlgorithm has to be "
                               "called before using get_ranking")
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        query = self.sentence_embedder.encode_sentence(query)
        if query is None:
            relevant_docs = []
        else:
            relevant_docs = self.document_lookup.similar_by_vector(query, topn=1000000)
        return pd.DataFrame(relevant_docs, columns=["id", "score"])


class PreTrainedEmbedding:
    def __init__(self, pretrained_model_path):
        self.model = None
        self.pretrained_model_path = pretrained_model_path

    def train_model(self):
        if self.model is None:
            self.model = load_facebook_vectors(self.pretrained_model_path)
            self.model.init_sims(True, True)
        return self.model


class FineTunedEmbedding:
    def __init__(self, pretrained_model_path, corpus: Corpus):
        self.model = None
        self.pretrained_model_path = pretrained_model_path
        self.corpus = corpus

    def train_model(self):
        if self.model is not None:
            return self.model
        self.model = load_facebook_model(self.pretrained_model_path)
        self.model.min_count = 1
        self.model.build_vocab(sentences=self.corpus.data, update=True)
        self.model.train(
            sentences=self.corpus.data,
            total_examples=len(self.corpus.data),
            epochs=10)
        self.model = self.model.wv
        self.model.init_sims(True, True)
        return self.model


class NewlyTrainedEmbedding:
    def __init__(self, corpus: Corpus, window_size, embedding_size, sg, negative):
        self.negative = negative
        self.model = None
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.sg = sg
        self.corpus = corpus

    def train_model(self):
        if self.model is not None:
            return self.model
        model = FastText(sg=self.sg,
                         size=self.embedding_size,
                         min_count=2,
                         window=self.window_size,
                         negative=self.negative)
        model.build_vocab(sentences=self.corpus.data)
        model.train(sentences=self.corpus.data,
                    total_examples=len(self.corpus.data), epochs=10)
        self.model = model.wv
        self.model.init_sims(True, True)
        return self.model

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

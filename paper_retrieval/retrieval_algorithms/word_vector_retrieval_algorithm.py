import logging
from typing import List

import numpy as np
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.fasttext import load_facebook_vectors, load_facebook_model, FastText
from gensim.models.keyedvectors import FastTextKeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from preprocessing import Corpus, apply_pipeline
from retrieval_algorithms import RetrievalAlgorithm
from retrieval_algorithms.identity import identity


class WordVectorRetrievalAlgorithm(RetrievalAlgorithm):
    def __init__(self, embedding, sentence_embedder):
        self.pipeline = None
        self.ids = None
        self.embedding = embedding
        self.sentence_embedder = sentence_embedder
        self.model = None
        self.doc_vectors = None

    def prepare(self, corpus: Corpus):
        self.pipeline = corpus.pipeline
        self.ids = pd.Series(corpus.ids, name="id")
        self.model = self.embedding.train_model(corpus)
        self.sentence_embedder = self.sentence_embedder(self.model, corpus)
        doc_vectors = corpus.data.progress_apply(self.sentence_embedder.encode_sentence)
        self.doc_vectors = normalize(np.stack(doc_vectors.values))

    def get_ranking(self, query: str) -> pd.DataFrame:
        if self.ids is None:
            raise RuntimeError("Prepare of class WordVectorRetrievalAlgorithm has to be "
                               "called before using get_ranking")
        query = apply_pipeline(query, self.pipeline)
        query = query.split(" ")
        query_vector = self.sentence_embedder.encode_sentence(query)
        if query_vector is None:
            return pd.DataFrame([], columns=["id", "score"])
        else:
            query_vector = normalize(query_vector.reshape(1, -1)).squeeze()
            return pd.DataFrame({
                "id": self.ids,
                "score": self.doc_vectors.dot(query_vector)
            }).sort_values(by="score", ascending=False)


class EpochLogger(CallbackAny2Vec):
    def __init__(self, name):
        self.epoch = 0
        self.name = name

    def on_epoch_begin(self, model):
        logging.info("\"{}\" epoch #{} start".format(self.name, self.epoch))

    def on_epoch_end(self, model):
        self.epoch += 1


class PreTrainedEmbedding:
    def __init__(self, pretrained_model_path: str):
        self.model = None
        self.pretrained_model_path = pretrained_model_path

    def train_model(self, corpus):
        if self.model is None:
            logging.info(f"Start loading model {self.pretrained_model_path}")
            if self.pretrained_model_path.endswith(".bin"):
                self.model = load_facebook_vectors(self.pretrained_model_path)
            else:
                self.model = FastTextKeyedVectors.load(self.pretrained_model_path)
            self.model.init_sims(True)
            logging.info(f"Finished loading model {self.pretrained_model_path}")
        return self.model


class FineTunedEmbedding:
    def __init__(self, pretrained_model_path, epochs):
        self.model = None
        self.pretrained_model_path = pretrained_model_path
        self.epochs = epochs

    def train_model(self, corpus):
        if self.model is not None:
            return self.model
        logging.info(f"Start fine tuning model {self.pretrained_model_path}")
        self.model = load_facebook_model(self.pretrained_model_path)
        self.model.min_count = 1
        self.model.build_vocab(sentences=corpus.data, update=True)
        self.model.train(
            sentences=corpus.data,
            total_examples=len(corpus.data),
            callbacks=[EpochLogger("Finetuned")],
            epochs=self.epochs)
        self.model = self.model.wv
        self.model.init_sims(True)
        logging.info(f"Finished fine tuning model {self.pretrained_model_path}")
        return self.model


class NewlyTrainedEmbedding:
    def __init__(self, window_size, embedding_size, sg, negative, epochs):
        self.negative = negative
        self.model = None
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.sg = sg
        self.epochs = epochs

    def train_model(self, corpus):
        if self.model is not None:
            return self.model
        model_name = f"Fasttext(win_size={self.window_size}, " \
                     f"emb_size={self.embedding_size}, neg={self.negative}," \
                     f" epoch={self.epochs}, sg={self.sg})"
        logging.info(f"Start training model {model_name}")
        model = FastText(sg=self.sg,
                         size=self.embedding_size,
                         min_count=2,
                         window=self.window_size,
                         callbacks=[EpochLogger(model_name)],
                         negative=self.negative)
        model.build_vocab(sentences=corpus.data)
        model.train(sentences=corpus.data,
                    total_examples=len(corpus.data), epochs=self.epochs)
        self.model = model.wv
        self.model.init_sims(True)
        logging.info(f"Finished training model {model_name}")
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

import pickle

import pandas as pd

from preprocessing import apply_pipeline, Corpus
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.similarities import SparseMatrixSimilarity
from gensim.models import CoherenceModel


import logging
logger = logging.getLogger(__name__)

class LDARetrieval():
    def __init__(self, corpus: Corpus, num_topics: int):
        self.ids = corpus.ids
        self.pipeline = corpus.pipeline
        logger.info("Create Dictionary for corpus vocabulary")
        self.id2word = corpora.Dictionary(corpus.data)
        bow_corpus = [self.id2word.doc2bow(text) for text in corpus.data]

        lda_logger = logging.getLogger("gensim.models.ldamodel")
        lda_logger.setLevel("WARN")
        log_filter = LogFilter(["converged", "PROGRESS"])
        lda_logger.addFilter(log_filter)
        logger.info(f"Start training lda model with {num_topics} topics")
        self.lda_model = LdaModel(corpus=bow_corpus,
                                  id2word=self.id2word,
                                  num_topics=num_topics,
                                  random_state=100,
                                  update_every=1,
                                  chunksize=10000,
                                  passes=2,
                                  alpha='auto',
                                  per_word_topics=True)
        logger.info("Compute LDA vectors for all documents")
        lda_corpus = [self.lda_model.get_document_topics(doc) for doc in bow_corpus]
        logger.info("Create sparse similarity matrix")
        self.index = SparseMatrixSimilarity(lda_corpus, num_features=num_topics)

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.pipeline)
        query = self.id2word.doc2bow(query.split(" "))
        df = pd.DataFrame(self.ids)
        df["score"] = self.index[self.lda_model.get_document_topics(query)]
        df.sort_values(by="score", ascending=False, inplace=True)
        return df[df["score"] > 0]

    def get_coherence_score(self, corpus):
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=corpus.data,
                                             coherence="u_mass")
        return coherence_model_lda.get_coherence()

    def save(self, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> "LDARetrieval":
        with open(file_path, "rb") as file:
            return pickle.load(file)


class LogFilter(logging.Filter):
    def __init__(self, content):
        super().__init__()
        self.content = content

    def filter(self, record):
        return any(string in record.getMessage() for string in self.content)


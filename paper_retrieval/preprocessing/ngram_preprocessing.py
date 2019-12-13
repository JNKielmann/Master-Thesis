import pandas as pd
from gensim.models.phrases import Phrases, Phraser

class BigramPreprocessor:
    def __init__(self, threshold: float):
        self.name = f"bigrams{threshold}"
        self.phraser = None
        self.threshold = threshold

    def fit_corpus(self, corpus: pd.Series):
        corpus = list(corpus.str.split(" "))
        phrases = Phrases(corpus, min_count=1, threshold=self.threshold)
        self.phraser = Phraser(phrases)

    def __call__(self, text: str):
        if self.phraser is None:
            raise RuntimeError("BigramPreprocessor has not been fit on a corpus. "
                               "Call fit_corpus before using it.")
        tokens = self.phraser[text.split(" ")]
        return " ".join(tokens)


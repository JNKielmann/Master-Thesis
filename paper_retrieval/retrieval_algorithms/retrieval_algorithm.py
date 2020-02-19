import abc

import pandas as pd

from preprocessing import Corpus


class RetrievalAlgorithm(abc.ABC):
    @abc.abstractmethod
    def prepare(self, corpus: Corpus):
        pass

    @abc.abstractmethod
    def get_ranking(self, query: str) -> pd.DataFrame:
        pass

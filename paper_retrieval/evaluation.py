import numpy as np
import pandas as pd


def average_precision(ranked_ids, relevant_ids):
    ranks = pd.Series(np.arange(1, len(ranked_ids) + 1))
    relevant_ranks = ranks[list(ranked_ids.isin(relevant_ids))]
    precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks.values
    return np.sum(precisions) / len(relevant_ids)

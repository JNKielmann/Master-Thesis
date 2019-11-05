from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd


def average_precision(ranked_ids, relevant_ids):
    ranks = pd.Series(np.arange(1, len(ranked_ids) + 1))
    relevant_ranks = ranks[list(ranked_ids.isin(relevant_ids))]
    precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks.values
    return np.sum(precisions) / len(relevant_ids)

def mean_average_precision(model, test_data):
    ap_list = []
    for test_row in test_data:
        query = test_row["query"]
        relevant_ids = test_row["documents"]
        ranked_ids = model.get_ranked_documents(query)["id"]
        ap_list.append(average_precision(ranked_ids, relevant_ids))
    return np.mean(ap_list)

def mean_average_precision_parallel(model, test_data, n_jobs):
    pool = Pool(n_jobs)
    work_chunks = np.array_split(test_data, n_jobs)
    map_list = pool.map(partial(mean_average_precision, model), work_chunks)
    return np.mean(map_list)

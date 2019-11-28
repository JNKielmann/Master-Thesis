from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def average_precision(ranked_ids, relevant_ids):
    ranks = pd.Series(np.arange(1, len(ranked_ids) + 1))
    relevant_ranks = ranks[list(ranked_ids.isin(relevant_ids))]
    precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks.values
    return np.sum(precisions) / len(relevant_ids)


def average_precision_table(model, test_data, progress_callback=None):
    ap_list = []
    for test_row in test_data:
        query = test_row["query"]
        relevant_ids = test_row["documents"]
        ranked_ids = model.get_ranked_documents(query)["id"]
        ap = average_precision(ranked_ids, relevant_ids)
        ap_list.append((query, ap))
        if progress_callback:
            progress_callback()
    return pd.DataFrame.from_records(ap_list, columns=["query", "average precision"])


def mean_average_precision(model, test_data, progress_callback=None):
    ap_table = average_precision_table(model, test_data, progress_callback)
    return ap_table["average precision"].mean()


def mean_average_precision_parallel(model, test_data, n_jobs):
    work_chunks = np.array_split(test_data, n_jobs)
    with Pool(n_jobs) as pool:
        map_list = pool.map(partial(mean_average_precision, model), work_chunks)
    return np.mean(map_list)


def evaluate_model(model, test_sets):
    results = {}
    total = np.sum([len(test_set) for _, test_set in test_sets])
    with tqdm(total=total, ncols='50%') as progress:
        for test_set_name, test_set in test_sets:
            data = [{
                "query": keyword_info["keyword"],
                "documents": keyword_info["paper_ids"]
            } for keyword_info in test_set]
            mAP = mean_average_precision(model, data, progress.update)
            results[test_set_name] = mAP
    return results


def train_evaluate_model(model_info, test_sets):
    model_name, model_factory = model_info
    result = evaluate_model(model_factory(), test_sets)
    return pd.DataFrame.from_dict({model_name: result}, orient="index")


def train_evaluate_models(model_infos, test_sets, n_jobs=4):
    with Pool(n_jobs) as pool:
        train_eval = partial(train_evaluate_model, test_sets=test_sets)
        results = pool.map(train_eval, model_infos)
    return pd.concat(results)

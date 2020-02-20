from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import logging


def average_precision(ranked_ids, relevant_ids):
    ranks = pd.Series(np.arange(1, len(ranked_ids) + 1))
    relevant_ranks = ranks[list(ranked_ids.isin(relevant_ids))]
    precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks.values
    return np.sum(precisions) / len(relevant_ids)


def precision_at(n):
    def calc_precision_at_n(ranked_ids, relevant_ids):
        return np.sum(ranked_ids[:n].isin(relevant_ids)) / n

    return calc_precision_at_n


def r_prec(ranked_ids, relevant_ids):
    return precision_at(len(relevant_ids))(ranked_ids, relevant_ids)


def bpref(ranked_ids, relevant_ids):
    num_relevant = len(relevant_ids)
    correct_results = ranked_ids.isin(relevant_ids)
    num_wrong_above = np.cumsum(~correct_results)
    num_wrong_above = np.clip(num_wrong_above, 0, num_relevant)
    ranks = pd.Series(np.arange(1, len(ranked_ids) + 1))
    sum_terms = 1 - (num_wrong_above[correct_results] / num_relevant)
    return np.sum(sum_terms) / num_relevant


def calculate_metrics(model, test_data, metrics, progress_callback=None):
    results = []
    for test_row in test_data:
        query = test_row["keyword"]
        relevant_ids = test_row["paper_ids"]
        ranked_ids = model.get_ranking(query)["id"]
        result_row = [query]
        for metric_name, metric_func in metrics:
            result_row.append(metric_func(ranked_ids, relevant_ids))
        results.append(result_row)
        if progress_callback:
            progress_callback()
    columns = ["query"] + [metric[0] for metric in metrics]
    return pd.DataFrame.from_records(results, columns=columns)


def confidence_interval_95(series):
    return 1.960 * series.std() / np.sqrt(len(series))


def calculate_average_metrics(model, test_data, metrics, progress_callback=None):
    metrics = calculate_metrics(model, test_data, metrics, progress_callback)
    average_metrics = metrics.drop("query", axis=1).agg(
        ["mean", "std", confidence_interval_95])
    return average_metrics.round(3).to_dict()


# def mean_average_precision_parallel(model, test_data, n_jobs):
#     with Pool(n_jobs) as pool:
#         map_list = pool.map(partial(mean_average_precision, model), test_data)
#     return np.mean(map_list)

default_metrics = [
    ("p@5", precision_at(5)),
    ("p@10", precision_at(10)),
    ("p@20", precision_at(20)),
    ("R-prec", r_prec),
    ("mAP", average_precision),
    ("bpref", bpref),
]


def evaluate_model(model, test_sets, metrics=None):
    if metrics is None:
        metrics = default_metrics
    results = {}
    total = np.sum([len(test_set) for _, test_set in test_sets])
    with tqdm(total=total, ncols='50%') as progress:
        for test_set_name, test_set in test_sets:
            test_set_result = calculate_average_metrics(model, test_set, metrics,
                                                        progress.update)
            for metric_name, metric_value in test_set_result.items():
                result_text = f"{metric_value['mean']:.3f}±" \
                              f"{metric_value['confidence_interval_95']:.3f} " \
                              f"({metric_value['std']:.3f})"
                results[(test_set_name, metric_name)] = result_text
    return results


def train_evaluate_model(model_info, test_sets):
    model_name, model, corpus = model_info
    logging.info("Start training model " + model_name)
    model.prepare(corpus)
    logging.info("Start evaluating model " + model_name)
    result = evaluate_model(model, test_sets)
    logging.info(f"Finished processing model {model_name}")
    return pd.DataFrame.from_dict({model_name: result}, orient="index")


def train_evaluate_models(model_infos, test_sets, n_jobs=4):
    train_eval = partial(train_evaluate_model, test_sets=test_sets)
    if n_jobs == 1:
        results = map(train_eval, model_infos)
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(train_eval, model_infos)
    return pd.concat(results)

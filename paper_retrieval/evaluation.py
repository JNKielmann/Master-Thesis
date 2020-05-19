"""
This files contains functions for training and evaluating models on test datasets and
compute different metrics.
"""
import logging
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def average_precision(ranked_ids, relevant_ids):
    """
    Compute average precision score
    :param ranked_ids: Document ids in order ranked by the algorithm
    :param relevant_ids: List of relevant document ids
    """
    ranks = pd.Series(np.arange(1, len(ranked_ids) + 1))
    relevant_ranks = ranks[list(ranked_ids.isin(relevant_ids))]
    precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks.values
    return np.sum(precisions) / len(relevant_ids)


def recall_at(n):
    """
    Returns function that computes the recall at n
    """
    def calc_recall_at_n(ranked_ids, relevant_ids):
        return np.sum(ranked_ids[:n].isin(relevant_ids)) / len(relevant_ids)

    return calc_recall_at_n


def precision_at(n):
    """
    Returns function that computes the precision at n
    """
    def calc_precision_at_n(ranked_ids, relevant_ids):
        return np.sum(ranked_ids[:n].isin(relevant_ids)) / n

    return calc_precision_at_n


def r_prec(ranked_ids, relevant_ids):
    """
    Compute r-precision score
    :param ranked_ids: Document ids in order ranked by the algorithm
    :param relevant_ids: List of relevant document ids
    """
    return precision_at(len(relevant_ids))(ranked_ids, relevant_ids)


def bpref(ranked_ids, relevant_ids):
    """
    Compute bpref score
    :param ranked_ids: Document ids in order ranked by the algorithm
    :param relevant_ids: List of relevant document ids
    """
    num_relevant = len(relevant_ids)
    correct_results = ranked_ids.isin(relevant_ids)
    num_wrong_above = np.cumsum(~correct_results)
    num_wrong_above = np.clip(num_wrong_above, 0, num_relevant)
    sum_terms = 1 - (num_wrong_above[correct_results] / num_relevant)
    return np.sum(sum_terms) / num_relevant


ignored_ids = []


def calculate_metrics(model, test_data, metrics, progress_callback=None,
                      id_row="paper_ids"):
    """
    Evaluate one model on the provided test data and compute metrics
    :param model: Model used for evaluation
    :param test_data: Data to evaluate on
    :param metrics: List of evaluation metrics to compute
    :param progress_callback: Callback called after each iteration
    :param id_row: Name of the column containing the ids in test_data
    """
    results = []
    for test_row in test_data:
        query = test_row["keyword"]
        relevant_ids = [id for id in test_row[id_row] if id not in ignored_ids]
        ranked_ids = pd.Series([id for id in model.get_ranking(query)["id"]
                                if id not in ignored_ids])
        result_row = [query]
        for metric_name, metric_func in metrics:
            result_row.append(metric_func(ranked_ids, relevant_ids))
        results.append(result_row)
        if progress_callback:
            progress_callback()
    columns = ["query"] + [metric[0] for metric in metrics]
    return pd.DataFrame.from_records(results, columns=columns)


def confidence_interval_95(series):
    """
    Compute confidence interval at 95 %
    :param series: Sequence to compute confidence interval on
    """
    return 1.960 * series.std() / np.sqrt(len(series))


def calculate_average_metrics(model, test_data, metrics, progress_callback=None):
    """
    Compute average value of metrics
    :param model: Model used for evaluation
    :param test_data: Data to evaluate on
    :param metrics: List of evaluation metrics to compute
    :param progress_callback: Callback called after each iteration
    :return:
    """
    metrics = calculate_metrics(model, test_data, metrics, progress_callback)
    average_metrics = metrics.drop("query", axis=1).agg(
        ["mean", "std", "var", confidence_interval_95])
    return average_metrics.round(3).to_dict()

# Default metrics used if no other are specified
default_metrics = [
    ("p@5", precision_at(5)),
    ("p@10", precision_at(10)),
    ("p@20", precision_at(20)),
    ("R-prec", r_prec),
    ("mAP", average_precision),
    ("bpref", bpref),
]


def evaluate_model(model, test_sets, metrics=None):
    """
    Evaluate model on multiple test sets
    :param model: Model used for evaluation
    :param test_sets: List of evaluation datasets
    :param metrics: List of evaluation metrics to compute
    """
    if metrics is None:
        metrics = default_metrics
    results = {}
    total = np.sum([len(test_set) for _, test_set in test_sets])
    with tqdm(total=total, ncols='50%') as progress:
        for test_set_name, test_set in test_sets:
            test_set_result = calculate_average_metrics(model, test_set, metrics,
                                                        progress.update)
            for metric_name, metric_value in test_set_result.items():
                results[(test_set_name, metric_name, "avg")] = metric_value['mean']
#                 results[(test_set_name, metric_name, "var")] = metric_value['var']
                results[(test_set_name, metric_name, "err")] = metric_value['confidence_interval_95']
    return results


def train_evaluate_model(model_info, test_sets, train=True, metrics=None):
    """
    Train model and then evaluate it on test datasets
    :param model_info: Model with additional metadata
    :param test_sets: List of evaluation datasets
    :param train: If False, model training is skipped
    :param metrics: List of metrics to compute
    """
    model_name, model, corpus = model_info
    if train:
        logging.info("Start training model " + model_name)
        model.prepare(corpus)
        logging.info("Start evaluating model " + model_name)
    result = evaluate_model(model, test_sets,metrics)
    logging.info(f"Finished processing model {model_name}")
    return pd.DataFrame.from_dict({model_name: result}, orient="index")


def train_evaluate_models(model_infos, test_sets, n_jobs=4):
    """
    Train and evaluate a list of models on the test datasets
    :param model_infos: List of models with additional metadata
    :param test_sets: List of evaluation datasets
    :param n_jobs: Number of threads to use for concurrent training and evaluation
    """
    train_eval = partial(train_evaluate_model, test_sets=test_sets)
    if n_jobs == 1:
        results = map(train_eval, model_infos)
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(train_eval, model_infos)
    return pd.concat(results)


def evaluate_models(model_infos, test_sets, n_jobs=4, metrics=None):
    """
    Evaluates a list of models
    :param model_infos: List of models with additional metadata
    :param test_sets: List of evaluation datasets
    :param n_jobs: Number of threads to use for concurrent evaluation
    :param metrics: List of metrics to compute
    """
    evaluate = partial(train_evaluate_model, train=False, test_sets=test_sets, metrics=metrics)
    if n_jobs == 1:
        results = map(evaluate, model_infos)
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(evaluate, model_infos)
    return pd.concat(results)


def to_latex_table(eval_table):
    """
    Convert pandas DataFrame with evaluation results to latex code
    :param eval_table: DataFrame with evaluation results
    :return:
    """
    def process_row(r):
        result = ""
        r = r.reorder_levels([1, 0])
        avg = r["avg"].values
        result += row_name + (" & {:.3f}" * len(avg)).format(*avg) + "\\\\\n"
        err = r["err"].values
        result += (" & \\small{{±{:.3f}}}" * len(err)).format(*err) + "\\\\[0.15cm]\n"
        return result

    general_table = r"\textbf{general queries}\\" + "\n"
    specific_table = r"\textbf{specific queries}\\" + "\n"
    for row_name, row_data in eval_table.iterrows():
        general_table += process_row(row_data[row_data.index.levels[0][0]])
        specific_table += process_row(row_data[row_data.index.levels[0][1]])

    return general_table + "\\addlinespace\n" + specific_table


def author_ranking_to_latex(eval_table):
    """
    Convert pandas DataFRame with author ranking evaluation results to latex code
    :param eval_table: DataFrame with author ranking evaluation results
    """
    def process_row(r, r_name):
        avg = r.xs("avg", level=2).values
        return r_name + (" & {:.3f}" * len(avg)).format(*avg) + "\\\\\n"

    latex_string = ""
    for row_name, row_data in eval_table.iterrows():
        latex_string += process_row(row_data, row_name)
    return latex_string

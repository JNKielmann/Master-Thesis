import copy
import pickle
from multiprocessing.pool import Pool
from time import time
from contextlib import suppress

import numpy as np
import pandas as pd
import os

import logging

logger = logging.getLogger(__name__)


def apply_pipeline(text, pipeline):
    for stage in pipeline:
        text = stage(text)
    return text


class Corpus:
    def __init__(self, file_path, preprocessing_pipeline, load_from_cache=True,
                 id_column="id", text_column="text", n_jobs=8):
        logger.info(f"Start preprocessing pipeline "
                    f"\"{_get_pipeline_name(preprocessing_pipeline)}\" "
                    f"for file {file_path}.")
        self.pipeline = copy.deepcopy(preprocessing_pipeline)
        cache_file_path = self._get_cache_file_path(file_path)
        if load_from_cache:
            if self._load_from_file(cache_file_path, id_column, text_column):
                logger.info(f"Loaded cached preprocessed corpus from {cache_file_path}")
                return
        df = pd.read_csv(file_path).fillna("")
        self.ids = df[id_column]
        self.data = df[text_column]
        pool = None
        for stage in self.pipeline:
            logger.info(f"Start stage \"{stage.name}\"")
            start_time = time()
            with suppress(AttributeError):
                stage.fit_corpus(self.data)
            if n_jobs > 1:
                if pool is None:
                    pool = Pool(n_jobs)
                self.data = pd.concat(
                    pool.map(ApplyStage(stage), np.array_split(self.data, n_jobs)))
            else:
                self.data = self.data.progress_apply(stage)
            logger.info(f"Finished stage \"{stage.name}\" in "
                        f"{time() - start_time:.2f} seconds")
        if pool is not None:
            pool.close()
        self._save_to_file(cache_file_path)
        self.data = self.data.str.split(" ")
        logger.info(f"Finished preprocessing pipeline. "
                    f"Saved preprocessed corpus to cache file {cache_file_path}")

    def _save_to_file(self, file_path):
        pd.concat([self.ids, self.data], axis=1).to_csv(file_path + ".csv")
        with open(file_path + ".pipeline", "wb") as file:
            pickle.dump(self.pipeline, file)

    def _load_from_file(self, file_path, id_column, text_column):
        data_exists = os.path.isfile(file_path + ".csv")
        pipeline_exists = os.path.isfile(file_path + ".pipeline")
        if not data_exists or not pipeline_exists:
            return False
        df = pd.read_csv(file_path + ".csv")
        self.ids = df[id_column]
        self.data = df[text_column].str.split(" ")
        with open(file_path + ".pipeline", "rb") as file:
            self.pipeline = pickle.load(file)
        return True

    def _get_cache_file_path(self, original_file_path):
        pipeline_name = _get_pipeline_name(self.pipeline)
        root, ext = os.path.splitext(original_file_path)
        return root + "_" + pipeline_name


def _get_pipeline_name(pipeline):
    return "_".join([stage.name for stage in pipeline])


class ApplyStage:
    def __init__(self, stage):
        self.stage = stage

    def __call__(self, data):
        return data.progress_apply(self.stage)

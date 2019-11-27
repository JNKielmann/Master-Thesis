import pickle
from functools import reduce

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import apply_pipeline


class QueryExpansionRetrieval:
    def __init__(self, wrapped_model, expansion_hierarchy,
                 only_expand_once, separate_weighting):
        self.wrapped_model = wrapped_model
        self.separate_weighting = separate_weighting
        self.only_expand_once = only_expand_once
        self.keyword_to_id = expansion_hierarchy["keyword_to_id"]
        self.keyword_to_id = {
            apply_pipeline(key, self.wrapped_model.pipeline): value
            for key, value in self.keyword_to_id.items()
        }
        self.id_to_children = expansion_hierarchy["id_to_children"]

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.wrapped_model.pipeline)
        expansion_terms = self.get_expansion_sub_terms(query, self.only_expand_once)
        if self.separate_weighting:
            return self._separate_weighting(query, expansion_terms)
        else:
            return self._same_weighting(query, expansion_terms)

    def _same_weighting(self, query, expansion_terms):
        expanded_query = query + " " + " ".join(expansion_terms)
        return self.wrapped_model.get_ranked_documents(expanded_query)

    def _separate_weighting(self, query, expansion_terms):
        expanded_query = " ".join(expansion_terms)
        normal_results = self.wrapped_model.get_ranked_documents(query)
        expanded_results = self.wrapped_model.get_ranked_documents(expanded_query)
        normal_results["inverse_rank"] = (1 / (
                    normal_results.reset_index().index + 1)) * 1
        expanded_results["inverse_rank"] = (1 / (
                expanded_results.reset_index().index + 1)) * 0.0
        merged_results = pd.merge(normal_results, expanded_results, on="id", how="outer")
        merged_results = merged_results.fillna(0)
        merged_results["score"] = merged_results["inverse_rank_x"] + merged_results[
            "inverse_rank_y"]
        merged_results.sort_values(by="score", ascending=False, inplace=True)
        return merged_results[["id", "score"]]

    def get_expansion_sub_terms(self, super_term, only_expand_once):
        keyword_ids = set()
        queue = []
        if super_term in self.keyword_to_id:
            keyword_id = self.keyword_to_id[super_term]
            queue += self.id_to_children[keyword_id]["child_ids"]
            while len(queue) > 0:
                keyword_id = queue.pop(0)
                keyword_data = self.id_to_children[keyword_id]
                keyword_ids.add(keyword_id)
                if not only_expand_once:
                    queue += [id for id in keyword_data["child_ids"] if
                              id not in keyword_ids]
        # result = set()
        # for kw_id in keyword_ids:
        #     for term in self.id_to_children[kw_id]["value"].split(" "):
        #         result.add(term)
        # return list(result)
        return [self.id_to_children[keyword_id]["value"] for keyword_id in keyword_ids]

    def save(self, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> "QueryExpansionRetrieval":
        with open(file_path, "rb") as file:
            return pickle.load(file)


def identity(x):
    return x

import pickle
from functools import reduce

import pandas as pd

from preprocessing import apply_pipeline


class QueryExpansionRetrieval:
    def __init__(self, wrapped_model, expansion_hierarchy):
        self.wrapped_model = wrapped_model
        self.keyword_to_id = expansion_hierarchy["keyword_to_id"]
        self.keyword_to_id = {
            apply_pipeline(key, self.wrapped_model.pipeline): value
            for key, value in self.keyword_to_id.items()
        }
        self.id_to_children = expansion_hierarchy["id_to_children"]

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        query = apply_pipeline(query, self.wrapped_model.pipeline)
        expansion_terms = self.get_expansion_sub_terms(query)
        expanded_query = query + " " + " ".join(expansion_terms)
        # print(expanded_query)
        # print()
        return self.wrapped_model.get_ranked_documents(expanded_query)

    def get_expansion_sub_terms(self, super_term):
        keyword_ids = set()
        queue = []
        if super_term in self.keyword_to_id:
            keyword_id = self.keyword_to_id[super_term]
            queue += self.id_to_children[keyword_id]["child_ids"]
            while len(queue) > 0:
                keyword_id = queue.pop(0)
                keyword_data = self.id_to_children[keyword_id]
                keyword_ids.add(keyword_id)
                queue += [id for id in keyword_data["child_ids"] if id not in keyword_ids]
        return [self.id_to_children[keyword_id]["value"] for keyword_id in keyword_ids]

    def save(self, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> "QueryExpansionRetrieval":
        with open(file_path, "rb") as file:
            return pickle.load(file)

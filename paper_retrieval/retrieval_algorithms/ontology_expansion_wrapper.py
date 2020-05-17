import pandas as pd

from preprocessing import Corpus
from retrieval_algorithms import RetrievalAlgorithm


class OntologyExpansionWrapper(RetrievalAlgorithm):

    def __init__(self, retrieval_algorithm: RetrievalAlgorithm, expansion_hierarchy,
                 only_expand_once, expansion_weight):
        self.expansion_weight = expansion_weight
        self.retrieval_algorithm = retrieval_algorithm
        self.only_expand_once = only_expand_once
        self.keyword_to_id = {}
        self.expansion_hierarchy = expansion_hierarchy
        self.id_to_children = expansion_hierarchy["id_to_children"]

    def prepare(self, corpus: Corpus):
        self.retrieval_algorithm.prepare(corpus)
        for keyword, kw_id in self.expansion_hierarchy["keyword_to_id"].items():
            processed_keyword = keyword.lower()
            if processed_keyword not in self.keyword_to_id:
                self.keyword_to_id[processed_keyword] = kw_id

    def get_ranking(self, query: str) -> pd.DataFrame:
        query = query.lower()
        expansion_terms = self.get_expansion_sub_terms(query, self.only_expand_once)
        return self._separate_weighting(query, expansion_terms)

    def _separate_weighting(self, query, expansion_terms):
        normal_results = self.retrieval_algorithm.get_ranking(query)
        if len(expansion_terms) == 0:
            return normal_results
        expanded_results = self.retrieval_algorithm.get_ranking(expansion_terms)
        joined_documents = pd.merge(normal_results, expanded_results, on="id",
                                    how="outer")
        joined_documents = joined_documents.fillna(0)
        score_x = self.expansion_weight * joined_documents["score_x"]
        score_y = (1 - self.expansion_weight) * joined_documents["score_y"]
        joined_documents["score"] = (score_x + score_y)
        joined_documents.sort_values(by="score", ascending=False, inplace=True)
        return joined_documents[["id", "score"]]
        # normal_results["inverse_rank"] = (1 / (
        #             normal_results.reset_index().index + 1)) * 1
        # expanded_results["inverse_rank"] = (1 / (
        #         expanded_results.reset_index().index + 1)) * 0.5
        # merged_results = pd.merge(normal_results, expanded_results, on="id", how="outer")
        # merged_results = merged_results.fillna(0)
        # merged_results["score"] = merged_results["inverse_rank_x"] + merged_results[
        #     "inverse_rank_y"]
        # merged_results.sort_values(by="score", ascending=False, inplace=True)
        # return merged_results[["id", "score"]]

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

import pandas as pd


class EnsembleRetrieval:
    def __init__(self, model1, model2, alpha):
        self.model1 = model1
        self.model2 = model2
        self.alpha = alpha

    def get_ranked_documents(self, query: str) -> pd.DataFrame:
        model1_scores = self.model1.get_ranked_documents(query)
        model2_scores = self.model2.get_ranked_documents(query)
        result = pd.merge(model1_scores, model2_scores, on="id", how="outer").fillna(0)
        result["score"] = self.alpha * result["score_x"] + result["score_y"]
        result.sort_values(by="score", inplace=True, ascending=False)
        return result[["id", "score"]]

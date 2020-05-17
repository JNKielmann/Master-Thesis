from typing import List

from expert_voting.author_score_modifiers.author_score_modifier import AuthorScoreModifier
from expert_voting.scored_expert import ScoredExpert


class CoAuthorModifier(AuthorScoreModifier):
    def __init__(self, doc_info, alpha):
        self.alpha = alpha
        self.co_author_matrix = {}
        for paper in doc_info.values():
            for author in paper["authors"]:
                if str(author["id"]) not in self.co_author_matrix:
                    self.co_author_matrix[str(author["id"])] = {}
                for co_author in paper["authors"]:
                    if co_author["id"] != author["id"]:
                        co_authors = self.co_author_matrix[str(author["id"])]
                        if str(co_author["id"]) not in co_authors:
                            co_authors[str(co_author["id"])] = 0
                        co_authors[str(co_author["id"])] += 1

    def modify(self, input_list: List[ScoredExpert]) -> List[ScoredExpert]:
        result = []
        input_list = list(sorted(input_list, key=lambda x: x.score, reverse=True))
       
        for author in input_list:
            additional_score = 0
            C = 0
            for co_author in input_list[:100]:
                if co_author.id != author.id:
                    co_author_papers = self.co_author_matrix[str(author.id)].get(
                        str(co_author.id), 0)
                    additional_score += co_author_papers * co_author.score
                    C += co_author_papers
            new_score = author.score + ((self.alpha * (additional_score/C)) if C>0 else 0)
            result.append(author.change_score(new_score))
        return result
import logging
import json
import requests
from requests import HTTPError
from typing import List, Dict, Iterator
from datetime import datetime

from pymongo import MongoClient
from tqdm import tqdm

from MAG_model import Author, Keyword, Journal, Conference, Paper

mag_api_url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate"


class ParsingError(Exception):
    pass


def abstract_from_inverted_index(word_count: int, inverted_index: Dict[str, List[int]]):
    word_list = [""] * word_count
    for word, occurrences in inverted_index.items():
        for occurrence in occurrences:
            word_list[occurrence] = word
    return " ".join(word_list)


def json_to_paper(entity, institute_name="karlsruhe institute of technology"):
    try:
        extended_data = json.loads(entity["E"])
        abstract = None
        if "IA" in extended_data:
            abstract = abstract_from_inverted_index(extended_data["IA"]["IndexLength"],
                                                    extended_data["IA"]["InvertedIndex"])
        return Paper(
            id=str(entity["Id"]),
            normalized_title=entity["Ti"],
            display_title=extended_data["DN"],
            publication_date=datetime.fromisoformat(entity["D"]),
            citation_count=entity["CC"],
            total_author_count=len(entity["AA"]),
            kit_authors=[Author(str(a["AuId"]), a["AuN"], a["S"]) for a in entity["AA"] if
                         a.get("AfN") == institute_name],
            keywords=[Keyword(str(k["FId"]), k["FN"]) for k in entity["F"]],
            journal=Journal(str(entity["J"]["JId"]),
                            entity["J"]["JN"]) if "J" in entity else None,
            conference=Conference(str(entity["C"]["CId"]),
                                  entity["C"]["CN"]) if "C" in entity else None,
            abstract=abstract
        )
    except ValueError as e:
        raise ParsingError(
            f"Paper with id {entity.get('Id', 'Unknown')} has invalid date: {entity['D']}") from e
    except KeyError as e:
        raise ParsingError(
            f"Paper with id {entity.get('Id', 'Unknown')} has missing key: {e}") from e


def request_papers(offset, page_size, institute_name="karlsruhe institute of technology"):
    try:
        response = requests.get(
            mag_api_url,
            params={
                "expr": "Composite(AA.AfN=='karlsruhe institute of technology')",
                "model": "latest",
                "count": page_size,
                "offset": offset,
                "attributes": "Id,Ti,D,CC,AA.AuN,AA.AuId,AA.AfN,AA.S,F.FN,F.FId,J.JN,J.JId,C.CN,C.CId,E"
            },
            headers={
                "Ocp-Apim-Subscription-Key": ""
            }
        )
        response.raise_for_status()
        return response.json()
    except HTTPError as http_err:
        raise RuntimeError(
            f"Error while requesting papers with offset {offset} for institute {institute_name}") from http_err


def request_all_papers(
        page_size,
        institute_name="karlsruhe institute of technology") -> Iterator[Paper]:
    offset = 0
    while True:
        papers_json = request_papers(offset, page_size, institute_name)
        offset += page_size
        if len(papers_json["entities"]) == 0:
            return
        for paper_entity in papers_json["entities"]:
            try:
                yield json_to_paper(paper_entity, institute_name)
            except ParsingError:
                logging.exception(
                    f"Error while trying to parse paper from json. Ignoring...")

if __name__ == '__main__':
    mongo_client = MongoClient("localhost", 27017, username="dev", password="Password4DEV")
    db = mongo_client["expert_recommender"]
    paper_collection = db["paper_collection"]
    for paper_json in tqdm((paper.to_json() for paper in request_all_papers(page_size=500))):
        paper_collection.insert_one(paper_json)

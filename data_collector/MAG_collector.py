import sys

import json
import requests
from requests import HTTPError
from typing import List, Dict, Iterator
from datetime import datetime
import time

from pymongo import MongoClient
from tqdm import tqdm

from MAG_model import Author, Keyword, Journal, Conference, Paper
import config

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

mag_api_url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate"


class ParsingError(Exception):
    pass


def abstract_from_inverted_index(word_count: int, inverted_index: Dict[str, List[int]]):
    word_list = [""] * word_count
    for word, occurrences in inverted_index.items():
        for occurrence in occurrences:
            word_list[occurrence] = word
    return " ".join(word_list)


def json_to_paper(entity):
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
                         a.get("AfN") == "karlsruhe institute of technology"],
            all_authors=[Author(str(a["AuId"]), a["AuN"], a["S"]) for a in entity["AA"]],
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


def request_papers(offset, page_size, expr):
    for i in range(6):
        response = requests.get(
            mag_api_url,
            params={
                "expr": expr,
                "model": "latest",
                "count": page_size,
                "offset": offset,
                "attributes": "Id,Ti,D,CC,AA.AuN,AA.AuId,AA.AfN,AA.S,F.FN,F.FId,J.JN,J.JId,C.CN,C.CId,E"
            },
            headers={
                "Ocp-Apim-Subscription-Key": config.mag_subscription_key
            }
        )
        try:
            response.raise_for_status()
            time.sleep(1)
            return response.json()
        except HTTPError as http_err:
            logging.exception(
                f"Error while requesting papers with offset {offset}"
                f" Response was: {response.text}")
            time.sleep((i + 1) * 10)
    raise RuntimeError("Failed to request papers 6 times")


def request_all_papers(page_size, expr) -> Iterator[Paper]:
    offset = 0
    while True:
        papers_json = request_papers(offset, page_size, expr)
        offset += page_size
        if len(papers_json["entities"]) == 0:
            return
        for paper_entity in papers_json["entities"]:
            try:
                yield json_to_paper(paper_entity)
            except ParsingError:
                logging.exception(
                    f"Error while trying to parse paper from json. Ignoring...")


def request_keywords(level, offset, page_size):
    for i in range(6):
        response = requests.get(
            mag_api_url,
            params={
                "expr": f"FL={level}",
                "model": "latest",
                "count": page_size,
                "offset": offset,
                "attributes": "Id,DFN,FL,FC.FId,FP.FId"
            },
            headers={
                "Ocp-Apim-Subscription-Key": config.mag_subscription_key
            }
        )
        try:
            response.raise_for_status()
            time.sleep(1)
            return response.json()
        except HTTPError as http_err:
            logging.exception(
                f"Error while requesting keywords with level {level} and offset {offset}."
                f" Response was: {response.text}")
            time.sleep((i + 1) * 10)
    raise RuntimeError("Failed to request keywords 6 times")


def request_all_keywords(page_size):
    for level in range(0, 6):
        offset = 0
        logging.info(f"Start requesting keywords for level {level}")
        while True:
            keyword_json = request_keywords(level, offset, page_size)
            offset += page_size
            if len(keyword_json["entities"]) == 0:
                break
            for keyword_entity in keyword_json["entities"]:
                try:
                    keyword = {
                        "id": str(keyword_entity["Id"]),
                        "value": keyword_entity["DFN"],
                        "level": keyword_entity["FL"],
                    }
                    if "FC" in keyword_entity:
                        keyword["childIds"] = [child["FId"] for child in
                                               keyword_entity["FC"]]
                    if "FP" in keyword_entity:
                        keyword["parentIds"] = [parent["FId"] for parent in
                                                keyword_entity["FP"]]
                    yield keyword
                except KeyError:
                    logging.exception(
                        f"Error while trying to parse keyword from json. Ignoring...")


def main():
    mongo_client = MongoClient("localhost", 27017, username=config.mongo_user,
                               password=config.mongo_password)
    db = mongo_client["expert_recommender"]

    # kit_paper_collection = db["paper_collection"]
    # institute_name = "karlsruhe institute of technology"
    # kit_papers = request_all_papers(page_size=1000,
    #                                 expr=f"Composite(AA.AfN=='{institute_name}')")
    # for paper_json in tqdm((paper.to_json() for paper in kit_papers)):
    #     kit_paper_collection.insert_one(paper_json)

    general_paper_collection = db["general_paper_collection"]
    general_papers = request_all_papers(page_size=500, expr="Y=2012")
    for paper_json in tqdm((paper.to_json() for paper in general_papers)):
        general_paper_collection.insert_one(paper_json)

    # keyword_collection = db["keywords_hierarchy"]
    # for keyword in tqdm(request_all_keywords(1000), total=665677):
    #     keyword_collection.insert_one(keyword)


if __name__ == '__main__':
    main()

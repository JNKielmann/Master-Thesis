import json
import logging
import sys
import time
from datetime import datetime
from typing import List, Dict, Iterator

import editdistance
import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from requests import HTTPError
from tqdm import tqdm

import config
from MAG_model import Author, Keyword, Paper, CitationContext

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

MAG_API_URL = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate"


class ParsingError(Exception):
    pass


def mag_request_page(offset, page_size, expr, attributes, num_tries=6):
    """
    Makes MAG Rest request to get page of results
    :param offset: Number of entries to skip
    :param page_size: Number of entries in this page
    :param expr: Query describing what information to get from the MAG
    :param attributes: List of attributes to return for the found MAG nodes
    :param num_tries: Number of retries before failing
    :return: MAG response
    """
    for i in range(num_tries):
        response = None
        try:
            response = requests.get(
                MAG_API_URL,
                params={
                    "expr": expr,
                    "model": "latest",
                    "count": page_size,
                    "offset": offset,
                    "attributes": attributes
                },
                headers={
                    "Ocp-Apim-Subscription-Key": config.mag_subscription_key
                }
            )
            response.raise_for_status()
            time.sleep(0.2)
            return response.json()
        except Exception as http_err:
            logging.exception(
                f"Error mag request {expr} error with offset {offset}"
                f" Response was: {response.text if response else ''}")
            time.sleep((i + 1) * 10)
    raise RuntimeError(f"Failed mag request {num_tries} times")


def mag_request_all(page_size, expr, attributes) -> Iterator[Paper]:
    """
    Request all items from the MAG for the query
    :param page_size: Number of items to fetch in each page
    :param expr: Query describing what information to get from the MAG
    :param attributes: List of attributes to return for the found MAG nodes
    :return: List of all items found
    """
    offset = 0
    while True:
        json_response = mag_request_page(offset, page_size, expr, attributes)
        offset += page_size
        yield from json_response["entities"]
        if len(json_response["entities"]) < page_size:
            return


def request_all_papers(page_size, expr, attributes) -> Iterator[Paper]:
    """
    Request all items from the MAG for the query
    :param page_size: Number of items to fetch in each page
    :param expr: Query describing what information to get from the MAG
    :param attributes: List of attributes to return for the found MAG nodes
    :return: List of all papers found
    """
    for paper_entity in mag_request_all(page_size, expr, attributes):
        try:
            yield json_to_paper(paper_entity)
        except ParsingError:
            logging.exception(
                f"Error while trying to parse paper from json. Ignoring...")


def abstract_from_inverted_index(word_count: int, inverted_index: Dict[str, List[int]]):
    """
    Create abstract string from MAG inverted intex
    :param word_count: Number of words in abstract
    :param inverted_index: The inverted index
    :return: Abstract as a string
    """
    word_list = [""] * word_count
    for word, occurrences in inverted_index.items():
        for occurrence in occurrences:
            word_list[occurrence] = word
    return " ".join(word_list)


def json_to_paper(entity):
    """
    Convert MAG json response to paper object
    :param entity: MAG json response
    :return: Paper object
    """
    try:
        extended_data = json.loads(entity["E"]) if "E" in entity else {}
        abstract = None
        if "IA" in extended_data:
            abstract = abstract_from_inverted_index(extended_data["IA"]["IndexLength"],
                                                    extended_data["IA"]["InvertedIndex"])
        authors = entity.get("AA", [])

        def json_to_author(author_entity):
            affiliation = author_entity.get("AfN", "")
            if affiliation == "":
                affiliation = author_entity.get("DAfN", "").lower()
            return Author(author_entity["AuId"],
                          author_entity["AuN"],
                          affiliation,
                          author_entity.get("S"))

        return Paper(
            id=str(entity["Id"]),
            normalized_title=entity.get("Ti"),
            display_title=extended_data.get("DN"),
            publication_date=datetime.fromisoformat(entity["D"]),
            citation_count=entity.get("CC"),
            total_author_count=len(authors),
            authors=[json_to_author(author_entity) for author_entity in authors],
            journal_conference_name=entity.get("VFN"),
            keywords=[Keyword(str(k["FId"]), k["FN"]) for k in entity.get("F", [])],
            referenced_paper_ids=[str(paper_id) for paper_id in entity.get("RId", [])],
            citation_contexts=[CitationContext(paper_id, citation_snippets) for
                               paper_id, citation_snippets in
                               entity.get("CitCon", {}).items()],
            abstract=abstract
        )
    except ValueError as e:
        raise ParsingError(
            f"Paper with id {entity.get('Id', 'Unknown')} has invalid date: "
            f"{entity['D']}") from e
    except KeyError as e:
        raise ParsingError(
            f"Paper with id {entity.get('Id', 'Unknown')} has missing key: {e}") from e


def request_keywords(level, offset, page_size):
    """
    Request field of study keywords from MAG for specific level
    :param level: Keywords level
    :param offset: Number of entries to skip
    :param page_size: Number of entries in this page
    :return: List of keywords
    """
    for i in range(6):
        response = requests.get(
            MAG_API_URL,
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
    """
    Request all field of study keywords from MAG
    :param page_size: Number of results per page
    :return: All keywords
    """
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


def save_kit_authors(kit_authors_collection):
    """
    Use heuristic to find KIT authors and save them to the MongoDB
    :param kit_authors_collection: MongoDB author collection
    """
    expr = f"Composite(AA.AfN='karlsruhe institute of technology')"
    attributes = "Id,AA.AuId,AA.AuN,AA.AfN,AA.DAfN,D"
    kit_papers = request_all_papers(1000, expr, attributes)
    kit_authors = {}
    for paper in tqdm(kit_papers, desc="requesting kit papers"):
        # author_counts = [(author_id, len(list(group))) for author_id, group in
        #                  groupby(paper.authors, lambda a: a.id)]
        # duplicate_authors = [counts[0] for counts in author_counts if counts[1] > 1]
        def has_valid_affiliation(author):
            kit = "karlsruhe institute of technology" in author.affiliation
            fzi = "center for information technology" in author.affiliation
            return kit or fzi

        paper_kit_authors = [author for author in paper.authors
                             if has_valid_affiliation(author)]
        for author in paper_kit_authors:
            if author.id not in kit_authors or \
                    kit_authors[author.id]["last_pub_date"] < paper.publication_date:
                kit_authors[author.id] = {
                    "last_pub_date": paper.publication_date,
                    "name": author.name
                }
    for author_id, author_data in tqdm(kit_authors.items(),
                                       desc="saving kit authors to mongo"):
        kit_authors_collection.insert_one({
            "id": author_id,
            "name": author_data["name"],
            "last_publication_date": author_data["last_pub_date"]
        })


def save_papers_of_kit_authors(kit_authors_collection, kit_paper_collection,
                               since_year=2019):
    """
    Save all papers written by authors at the KIT
    :param kit_authors_collection: MongoDB author collection
    :param kit_paper_collection: MongoDB paper collection
    :param since_year: The year in which a researcher as to have published a paper in
                       order to be considered an expert
    """
    kit_paper_collection.create_index("id", unique=True)
    relevant_authors = list(kit_authors_collection.find({
        "last_publication_date": {"$gte": datetime(since_year, 1, 1)}
    }))
    attributes = "Id,Ti,D,CC,AA.AuN,AA.AuId,AA.AfN," \
                 "AA.S,F.FN,F.FId,CitCon,RId,E"
    processed_paper_ids = set()
    for author in tqdm(relevant_authors[3413:], desc="Get papers for experts"):
        expr = f"Composite(AA.AuId={int(author['id'])})"
        author_papers = request_all_papers(1000, expr, attributes)
        for paper in author_papers:
            if paper.id not in processed_paper_ids:
                processed_paper_ids.add(paper.id)
                try:
                    kit_paper_collection.insert_one(paper.to_json())
                except DuplicateKeyError:
                    pass


def remove_duplicate_papers(kit_paper_collection):
    all_papers = list(kit_paper_collection.find({}, {"normalized_title": 1}))
    duplicate_paper_ids = set()

    def similar(text1, text2):
        return editdistance.eval(text1, text2) < (len(text1) * 0.2 + 1)
        # return text1 == text2

    for i in tqdm(range(27000, len(all_papers)), desc="Find duplicate papers",
                  total=len(all_papers), initial=27000):
        if all_papers[i]["_id"] in duplicate_paper_ids:
            continue
        paper_i_title = all_papers[i]["normalized_title"]
        for j in range(i + 1, len(all_papers)):
            paper_j_title = all_papers[j]["normalized_title"]
            if similar(paper_i_title, paper_j_title):
                duplicate_paper_ids.add(all_papers[j]["_id"])
                print("Duplicate: ")
                print(paper_i_title, all_papers[i]["_id"])
                print(paper_j_title, all_papers[j]["_id"])
                print(editdistance.eval(paper_i_title, paper_j_title))
                kit_paper_collection.delete_one({"_id": all_papers[j]["_id"]})
    print(f"Removed {len(duplicate_paper_ids)} duplicate papers")


def main():
    mongo_client = MongoClient("localhost", 27017, username=config.mongo_user,
                               password=config.mongo_password)
    db = mongo_client["expert_recommender"]

    kit_authors_collection = db["mag_kit_authors"]
    save_kit_authors(kit_authors_collection)

    save_papers_of_kit_authors(kit_authors_collection, db["kit_expert_papers"])
    remove_duplicate_papers(db["kit_expert_papers"])

    general_paper_collection = db["general_paper_collection"]
    general_papers = request_all_papers(page_size=500, expr="Y=2019")
    for paper_json in tqdm((paper.to_json() for paper in general_papers)):
        general_paper_collection.insert_one(paper_json)

    keyword_collection = db["keywords_hierarchy"]
    for keyword in tqdm(request_all_keywords(1000), total=665677):
        keyword_collection.insert_one(keyword)


if __name__ == '__main__':
    main()

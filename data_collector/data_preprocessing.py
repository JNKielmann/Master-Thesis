import re
import sys
from time import time
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import langid
from whatthelang import WhatTheLang
import config
import json

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def get_authors_since(collection, year):
    return list(collection.aggregate([
        {
            "$unwind": "$kit_authors"
        },
        {
            "$group": {
                "_id": "$kit_authors.id",
                "last_paper_date": {
                    "$max": "$publication_date"
                },
                "paper_count": {
                    "$sum": 1
                }
            }
        },
        {
            "$match": {
                "$and": [
                    {"paper_count": {"$gt": 1}},
                    {"last_paper_date": {"$gt": datetime(year, 1, 1)}}
                ]
            }
        },
        {
            "$group": {
                "_id": None,
                "author_ids": {"$push": "$_id"}
            }
        }
    ]))[0]["author_ids"]


def get_papers_of_authors(collection, author_ids):
    return collection.aggregate([
        {
            "$match": {
                "kit_authors.id": {"$in": author_ids}
            }
        },
        {
            "$project": {
                "id": 1,
                "display_title": 1,
                "abstract": 1,
                "keywords": 1,
                "kit_authors": {
                    "$filter": {
                        "input": "$kit_authors",
                        "as": "item",
                        "cond": {"$in": ["$$item.id", author_ids]}
                    }
                }
            }
        }
    ])


def get_expert_papers(db):
    start_time = time()
    logging.info("Start retrieving expert papers from MongoDB")
    collection = db["paper_collection"]
    authors = get_authors_since(collection, 2017)
    logging.info(f"Found {len(authors)} authors that wrote at least one KIT paper "
                 f"in the year 2017 or later. These are considered KIT experts.")
    papers = pd.DataFrame(get_papers_of_authors(collection, authors)).drop("_id", axis=1)
    logging.info(f"Found {len(papers)} papers written by KIT experts")
    logging.info(f"Finished retrieving papers in {time() - start_time:.2f} seconds")
    return papers


def get_keywords_for(db, paper_ids):
    return pd.DataFrame(db["paper_collection"].aggregate([
        {
            "$match": {
                "id": {"$in": paper_ids}
            }
        },
        {
            "$unwind": "$keywords"
        },
        {
            "$group": {
                "_id": "$keywords.id",
                "keyword": {"$first": "$keywords.value"},
                "paper_ids": {
                    "$push": "$id"
                }
            }
        },
        {
            "$lookup": {
                "from": "keywords",
                "localField": "_id",
                "foreignField": "id",
                "as": "keyword_data"
            }
        },
        {
            "$addFields": {
                "level": {"$arrayElemAt": ["$keyword_data.level", 0]},
                "keyword_id": "$_id"
            }
        },
        {
            "$project": {
                "keyword_data": 0,
                "_id": 0
            }
        }
    ]))


def filter_missing_abstracts(papers):
    filtered_papers = papers.dropna()
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers without abstracts")
    return filtered_papers


def filter_short_abstracts(papers, min_length):
    filtered_papers = papers[papers["abstract"].apply(len) >= min_length]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers with abstracts "
                 f"shorter than {min_length} characters")
    return filtered_papers


def filter_long_abstracts(papers, max_length):
    filtered_papers = papers[papers["abstract"].apply(len) <= max_length]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers with abstracts "
                 f"longer than {max_length} characters")
    return filtered_papers


def filter_non_english_abstracts(papers):
    wtl = WhatTheLang()
    abstracts = papers["abstract"].apply(lambda text: re.sub(r"[\r\n]+", " ", text))
    wtl_lang = abstracts.apply(wtl.predict_lang)
    langid_lang = abstracts.apply(lambda a: langid.classify(a)[0])
    filtered_papers = papers[(wtl_lang == "en") & (langid_lang == "en")]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers "
                 f"with non english abstracts")
    return filtered_papers


def filter_bad_papers(papers):
    start_time = time()
    logging.info("Start filtering out papers with bad abstracts")
    papers = filter_missing_abstracts(papers)
    papers = filter_short_abstracts(papers, 400)
    papers = filter_long_abstracts(papers, 4000)
    papers = filter_non_english_abstracts(papers)
    logging.info(f"Finished filtering papers in {time() - start_time:.2f} seconds")
    logging.info(f"Number of papers after filtering: {len(papers)}")
    return papers


def export_papers_csv(papers, path):
    logging.info(f"Writing papers to {path}")
    papers[["id", "text"]].to_csv(path, index=False)


def export_keywords_csv(keywords, path):
    logging.info(f"Writing keywords to {path}")
    keywords.to_csv(path, index=False)


def export_keywords_json(keywords, path):
    logging.info(f"Writing keywords to {path}")
    keywords.to_json(path, orient="records")


def get_keyword_hierarchy(db):
    result = {}
    keywords = list(db["keywords_hierarchy"].find())
    result["keyword_to_id"] = {
        keyword["value"]: str(keyword["id"])
        for keyword in keywords
    }
    result["id_to_children"] = {
        str(keyword["id"]): {
            "child_ids": [str(k) for k in keyword.get("childIds", [])],
            "value": keyword["value"]
        }
        for keyword in keywords
    }
    return result


def main():
    logging.info("Start preprocessing data")
    mongo_client = MongoClient("localhost", 27017,
                               username=config.mongo_user,
                               password=config.mongo_password)
    db = mongo_client["expert_recommender"]
    # expert_papers = get_expert_papers(db)
    # expert_papers = filter_bad_papers(expert_papers)
    # expert_papers["text"] = \
    #     expert_papers["display_title"] + " " + expert_papers["abstract"]
    # export_papers_csv(expert_papers, "../data/kit_expert_2017_papers.csv")
    # keywords_for_expert_papers = get_keywords_for(db, list(expert_papers["id"]))
    # export_keywords_json(keywords_for_expert_papers,
    #                      "../data/kit_expert_2017_keywords.json")
    keyword_hierarchy = get_keyword_hierarchy(db)
    with open("../data/keyword_hierarchy.json", 'w') as file:
        json.dump(keyword_hierarchy, file)

    logging.info("Finished preprocessing data")


if __name__ == '__main__':
    main()

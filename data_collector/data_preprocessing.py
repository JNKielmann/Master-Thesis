import itertools
import json
import logging
import re
import sys
from datetime import datetime
from time import time

import langid
import pandas as pd
from pymongo import MongoClient
from whatthelang import WhatTheLang

import config

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def get_expert_papers(paper_collection, author_collection):
    """
    Query papers written by KIT experts from MongoDB
    :param paper_collection: MongoDB paper collection
    :param author_collection: MongoDB author collection
    """
    start_time = time()
    logging.info("Start retrieving expert papers from MongoDB")
    experts = author_collection.find({
        "last_publication_date": {"$gte": datetime(2019, 1, 1)}
    }, {"id": 1})
    expert_ids = set([str(expert["id"]) for expert in experts])
    papers = list(paper_collection.find({}, {
        "referenced_paper_ids": 0,
        "keywords": 0,
        "citation_contexts": 0
    }))
    for paper in papers:
        paper["authors"] = [author for author in paper["authors"]
                            if str(author["id"]) in expert_ids]
    papers = pd.DataFrame(papers).drop("_id", axis=1)
    logging.info(f"Found {len(papers)} papers written by KIT experts")
    logging.info(f"Finished retrieving papers in {time() - start_time:.2f} seconds")
    return papers


def get_keyword_hierarchy(db):
    """
    Get the keyword hierarchy from the MongoDB
    :param db: MongoDB connection
    :return: Keyword hierarchy dict
    """
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


def get_keywords_for(paper_collection, paper_ids):
    """
    Get a list of keyword each paper has. Is used as evaluation dataset of retrieval algorithms
    :param paper_collection: MongoDB paper collection
    :param paper_ids: Ids of papers to get keywords for
    """
    keywords = list(paper_collection.aggregate([
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
                "from": "keywords_hierarchy",
                "localField": "_id",
                "foreignField": "id",
                "as": "keyword_data"
            }
        },
        {
            "$addFields": {
                "level": {"$arrayElemAt": ["$keyword_data.level", 0]},
                "keyword": {"$arrayElemAt": ["$keyword_data.value", 0]},
                "child_ids": {"$arrayElemAt": ["$keyword_data.childIds", 0]},
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
    # return keywords
    level = 0
    id_to_keyword = {k["keyword_id"]: k for k in keywords}
    while level >= 7:
        for keyword in [k for k in keywords if k["level"] == level]:
            for keyword_child_id in keyword.get("child_ids", []):
                child_keyword = id_to_keyword.get(str(keyword_child_id), None)
                if child_keyword is not None:
                    keyword["paper_ids"] += child_keyword["paper_ids"]
            keyword["paper_ids"] = list(set(keyword["paper_ids"]))
        level += 1
    return pd.DataFrame(keywords)


def filter_missing_abstracts(papers):
    """
    Remove papers without abstract
    :param papers: DataFrame with papers
    """
    filtered_papers = papers[~papers["abstract"].isna()]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers without abstracts")
    return filtered_papers


def filter_short_abstracts(papers, min_length):
    """
    Remove papers with short abstract
    :param papers: DataFrame with papers
    :param min_length: minimum abstract length to not be removed
    """
    filtered_papers = papers[papers["abstract"].apply(len) >= min_length]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers with abstracts "
                 f"shorter than {min_length} characters")
    return filtered_papers


def filter_long_abstracts(papers, max_length):
    """
    Remove papers with too long abstracts
    :param papers: DataFrame with papers
    :param min_length: minimum abstract length to not be removed
    :param max_length:
    """
    filtered_papers = papers[papers["abstract"].apply(len) <= max_length]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers with abstracts "
                 f"longer than {max_length} characters")
    return filtered_papers


def filter_non_english_abstracts(papers):
    """
    Remove abstracts that are not written in englisch
    :param papers: DataFrame with papers
    """
    wtl = WhatTheLang()
    abstracts = papers["abstract"].apply(lambda text: re.sub(r"[\r\n]+", " ", text))
    wtl_lang = abstracts.apply(wtl.predict_lang)
    langid_lang = abstracts.apply(lambda a: langid.classify(a)[0])
    filtered_papers = papers[(wtl_lang == "en") & (langid_lang == "en")]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers "
                 f"with non english abstracts")
    return filtered_papers


def filter_incorrect_abstract(papers):
    """
    Remove papers with blacklisted terms to remove incorrectly parsed abstracts
    :param papers: DataFrame with papers
    """
    lower_abstracts = papers["abstract"].str.lower()
    filtered_papers = papers[
        lower_abstracts.str.count("university") +
        lower_abstracts.str.count("department") +
        lower_abstracts.str.count("institut") +
        lower_abstracts.str.count("science foundation") +
        lower_abstracts.str.count("council") +
        lower_abstracts.str.count("academy") +
        lower_abstracts.str.count("school") < 3]
    filtered_papers = filtered_papers[
        filtered_papers["abstract"].str.count("\\.") /
        filtered_papers["abstract"].str.len() < 0.04]
    filtered_papers = filtered_papers[
        filtered_papers["abstract"].str.count("Dr\\.") < 2]
    logging.info(f"Removed {len(papers) - len(filtered_papers)} papers with abstracts "
                 f"that were incorrectly parsed by the MAG")
    return filtered_papers


def filter_bad_papers(papers):
    """
    Remove papers with low quality abstracts
    :param papers: DataFrame with papers
    """
    start_time = time()
    logging.info("Start filtering out papers with bad abstracts")
    papers = filter_missing_abstracts(papers)
    papers = filter_short_abstracts(papers, 400)
    papers = filter_long_abstracts(papers, 4000)
    papers = filter_non_english_abstracts(papers)
    papers = filter_incorrect_abstract(papers)
    logging.info(f"Finished filtering papers in {time() - start_time:.2f} seconds")
    logging.info(f"Number of papers after filtering: {len(papers)}")
    return papers


def export_paper_abstracts_csv(papers, path):
    """
    Save paper abstracts in csv file
    :param papers: Papers DataFrame
    :param path: Path of csv file
    """
    logging.info(f"Writing papers to {path}")
    columns = ["id", "text"]
    papers[columns].to_csv(path, index=False)


def export_paper_info_json(papers, path):
    """
    Save paper info data to json file
    :param papers: Papers DataFrame
    :param path: Path of paper info json file
    """
    papers["authors"] = papers["authors"].apply(
        lambda authors: [{
            "id": author["id"],
            "order_in_paper": author["order_in_paper"]
        } for author in authors]
    )
    columns = ["id", "display_title", "publication_date", "citation_count", "authors"]
    papers[columns].set_index("id").to_json(path, orient="index")


def has_valid_affiliation(author):
    kit = "karlsruhe institute of technology" in author["affiliation"]
    fzi = "center for information technology" in author["affiliation"]
    return kit or fzi


def get_author_kit_score(paper_collection, author_collection):
    """
    Count number of KIT and non-KIT papers
    """
    authors = list(author_collection.find({
        "last_publication_date": {"$gte": datetime(2019, 1, 1)}
    }))
    author_map = {author["id"]: {**author,
                                 "kit_paper": 0,
                                 "non_kit_paper": 0,
                                 "unknown_paper": 0}
                  for author in authors}
    papers = list(paper_collection.find({
        "publication_date": {"$gte": datetime(2019, 1, 1)}
    }))
    for paper in papers:
        author_counts = {author_id: len(list(group)) for author_id, group in
                         itertools.groupby(paper["authors"], lambda a: a["id"])}
        for author in paper["authors"]:
            if author["id"] in author_map:
                author_info = author_map[author["id"]]
                if has_valid_affiliation(author):
                    author_info["kit_paper"] += 1 / author_counts[author["id"]]
                elif author["affiliation"] != "":
                    author_info["non_kit_paper"] += 1 / author_counts[author["id"]]
                else:
                    author_info["unknown_paper"] += 1 / author_counts[author["id"]]
    return author_map


def export_author_info_json(paper_collection, author_collection, gs_author_collection,
                            path):
    """
    Save author info to json file
    """
    authors = list(author_collection.find({
        "last_publication_date": {"$gte": datetime(2019, 1, 1)}
    }, {
        "id": 1,
        "name": 1,
        "last_publication_date": 1
    }))
    author_info = get_author_kit_score(paper_collection, author_collection)
    for author in authors:
        gs_author = gs_author_collection.find_one({"name_normalized": author["name"]})
        if gs_author is not None:
            author["gs_keywords"] = gs_author["keywords"]
            author["gs_id"] = gs_author["id"]
    author_map = {author["id"]: {
        "name": author["name"],
        "last_publication_date": int(
            datetime.timestamp(author["last_publication_date"]) * 1000),
        "gs_keywords": author.get("gs_keywords", []),
        "gs_id": author.get("gs_id", None),
        "kit_paper": author_info[author["id"]]["kit_paper"],
        "non_kit_paper": author_info[author["id"]]["non_kit_paper"],
        "unknown_paper": author_info[author["id"]]["unknown_paper"],
    } for author in authors}
    with open(path, 'w') as file:
        json.dump(author_map, file)


def export_keywords_csv(keywords, path):
    logging.info(f"Writing keywords to {path}")
    keywords.to_csv(path, index=False)


def export_keywords_json(keywords, path):
    logging.info(f"Writing keywords to {path}")
    keywords.to_json(path, orient="records")


def get_citation_snippets(db):
    """
    Get citation snippets from MongoDB
    """
    citation_snippets = pd.DataFrame(db["kit_expert_papers"].aggregate([
        {
            "$unwind": "$citation_contexts"
        }, {
            "$group": {
                "_id": "$citation_contexts.cited_paper_id",
                "citation_context_snippets": {
                    "$push": "$citation_contexts.citation_snippets"
                }
            }
        }, {
            "$lookup": {
                "from": "kit_expert_papers",
                "localField": "_id",
                "foreignField": "id",
                "as": "kit_expert_paper"
            }
        }, {
            "$match": {
                "kit_expert_paper": {"$ne": []}
            }
        }, {
            "$project": {
                "id": "$_id",
                "_id": 0,
                "citation_context_snippets": 1
            }
        }
    ], allowDiskUse=True))
    citation_snippets["citation_context_snippets"] = citation_snippets[
        "citation_context_snippets"].apply(lambda x: " ".join(itertools.chain(*x)))
    return citation_snippets


def main():
    logging.info("Start preprocessing data")
    mongo_client = MongoClient("localhost", 27017,
                               username=config.mongo_user,
                               password=config.mongo_password)
    db = mongo_client["expert_recommender"]

    citation_snippets = get_citation_snippets(db)
    expert_papers = get_expert_papers(db["kit_expert_papers"], db["mag_kit_authors"])
    expert_papers = pd.merge(expert_papers, citation_snippets, on="id", how="left")

    expert_papers = filter_bad_papers(expert_papers)
    expert_papers = expert_papers.fillna("")

    expert_papers["text"] = \
        expert_papers["display_title"] + " " + expert_papers["abstract"]
    export_paper_abstracts_csv(expert_papers, "../data/kit_expert_2019_all_papers.csv")

    expert_papers["text"] = \
        expert_papers["display_title"] + " " + expert_papers["abstract"] + " " + \
        expert_papers["journal_conference_name"] + " " + expert_papers[
            "citation_context_snippets"]
    export_paper_abstracts_csv(expert_papers,
                               "../data/kit_expert_2019_all_papers_journal_citations.csv")

    export_paper_info_json(expert_papers, "../data/kit_expert_2019_all_paper_info.json")

    export_author_info_json(db["kit_expert_papers"], db["mag_kit_authors"],
                            db["gs_kit_authors"],
                            "../data/kit_expert_2019_all_author_info.json")

    keyword_hierarchy = get_keyword_hierarchy(db)
    with open("../data/keyword_hierarchy.json", 'w') as file:
        json.dump(keyword_hierarchy, file)

    keywords_for_expert_papers = get_keywords_for(db["kit_expert_papers"],
                                                  list(expert_papers["id"]))
    export_keywords_json(keywords_for_expert_papers,
                         "../data/kit_expert_2019_all_keywords.json")

    logging.info("Finished preprocessing data")

if __name__ == '__main__':
    main()

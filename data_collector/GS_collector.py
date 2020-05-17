import codecs
import itertools
import sys
import logging
import re
from time import sleep
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient

import config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

base_url = "https://scholar.google.com/"
# start_path = "/citations?view_op=view_org&hl=en&org=3515922183173499558"
start_path = "/citations?hl=en&view_op=search_authors&mauthors=kit&btnG="


class ParsingError(Exception):
    pass


def request_page(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_author(author_element):
    try:
        name_element = author_element.select_one(".gs_ai_name a")
        id = re.search("user=([^&]*)", name_element.get("href")).group(1)
        name = name_element.text
        keywords = [keyword_element.text for keyword_element in
                    author_element.select(".gs_ai_one_int")]
    except AttributeError as e:
        raise ParsingError("Error while parsing author html: \n" +
                           author_element.prettify(), e)
    return {
        "id": id,
        "name": name,
        "keywords": keywords
    }


def parse_authors(document):
    author_elements = document.select(".gsc_1usr")
    if len(author_elements) == 0:
        raise ParsingError("HTML contains no authors: \n" + document.prettify())
    authors = []
    for author_element in author_elements:
        try:
            authors.append(parse_author(author_element))
        except ParsingError as e:
            logging.error("Error parsing author. Skip to next one.", e)
    return authors


def parse_next_link(document):
    try:
        next_button_onclick = document.select_one(".gsc_pgn_pnx").get("onclick")
        link = re.search("location='([^']*)", next_button_onclick).group(1)
        link = codecs.decode(link, "unicode_escape")
        return link
    except Exception as e:
        logging.error("Error parsing next link:\n" + document.prettify(), e)
        return None


def process_page(url):
    html_text = request_page(url)
    document = BeautifulSoup(html_text, "html.parser")
    authors = parse_authors(document)
    next_link = parse_next_link(document)
    return authors, next_link


def execute_with_retry(func, retries):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logging.error(f"Error on try {i}", e)
            sleep(2 ** i * 10)
    raise RuntimeError(f"Execution failed {retries} times")


def main():
    mongo_client = MongoClient("localhost", 27017, username=config.mongo_user,
                               password=config.mongo_password)
    gs_kit_authors = mongo_client["expert_recommender"]["gs_kit_authors_2"]
    url = base_url + start_path
    for i in itertools.count():
        logging.info(f"Requesting page {i + 1} with url {url}")
        try:
            authors, next_link = execute_with_retry(lambda: process_page(url), 5)
        except RuntimeError:
            logging.error("Request failed for url " + url)
            break
        gs_kit_authors.insert_many(authors)
        if next_link is None:
            break
        url = base_url + next_link
        sleep(1)


if __name__ == '__main__':
    main()

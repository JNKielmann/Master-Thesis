import json
import pickle
import runpy

import spacy
from flask import Flask, jsonify, request, abort
from flask_cors import CORS, cross_origin

import sys

sys.path.insert(0, "..")

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
cors = CORS(app)

try:
    spacy.load('en', disable=['ner'])
except IOError:
    print("Downloading spacy model...")
    spacy.cli.download("en")

file_path = "../../data/models/expert_voting.model"
with open(file_path, "rb") as file:
    model = pickle.load(file)
with open("../../data/kit_expert_2019_all_author_info.json") as file:
    author_info = json.load(file)


@app.route("/experts", methods=["GET"])
@cross_origin()
def get_experts():
    if "query" not in request.args:
        return {"error": 'Query parameter "query" is required'}, 400
    query = request.args["query"]
    limit = request.args.get("limit", 15)
    result = model.get_ranking(query, author_info).head(limit)
    result = result[["id", "score", "name"]].to_dict(orient="records")
    return jsonify({"experts": result})


if __name__ == '__main__':
    app.run(port=8071, debug=False, use_reloader=False)

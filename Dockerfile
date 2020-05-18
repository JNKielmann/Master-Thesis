FROM python:3.7-buster
RUN pip install pipenv

COPY paper_retrieval/server/requirements.txt /
RUN pip install -r requirements.txt
RUN pip install -e git+https://github.com/epfml/sent2vec.git@c6e8aa7bbbf79af967bd7bb7ef6c0c315dd9cbc7#egg=sent2vec
RUN python -m spacy download en

COPY data/models /data/models
COPY data/kit_expert_2019_all_author_info.json /data/
COPY paper_retrieval/ /app
CMD python /app/server/flask_server.py

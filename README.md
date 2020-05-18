# Master thesis: Expert recommendation.
This repository contains the code for my master thesis "A Voting-Based Approach for Unsupervised Expert Recommendation on Bibliographic Data".

Abstract:
> Expert recommendation systems, which allow users to find relevant experts based on a few input keywords, help promote collaboration and reduce duplication of effort in large organizations.
In this work, an expert recommendation system for the KIT is developed, which finds relevant experts by analyzing bibliographic data.
A document-centric approach is chosen that first retrieves all relevant publications for a given input query and then uses these documents as evidence for the expertise of the authors who wrote them.
Multiple document retrieval algorithms, including BM25, query language models and document embeddings, are evaluated on a dataset of papers that were published at the KIT.
This dataset is constructed using data from the Microsoft Academic Graph. On top of the best performing document retrieval model, which is a combination of a BM25 and Sent2Vec model with query expansion, an expert voting algorithm is applied.
It works by allowing all relevant documents to vote for their authors. These votes can be weighted differently depending on the relevance score of the document and other metadata like paper publication dates and co-author information. 
This voting algorithm is tuned using a manually created validation dataset. It was found that using a simple voting function that sums the document relevance scores and incorporates the reciprocal rank of the documents forms a good baseline that can be slightly improved using the additional publication metadata. 
A final evaluation was done using data from the existing “Kompetenzpool” expert recommendation system and data collected from an online survey. The results show that the system is able to recommend most ex- perts that are returned by the “Kompetenzpool” or are named by the participants of the survey, without the need for manually created researcher profiles.

### Running the expert recommendation REST server
The expert recommendation server uses the final expert retrieval model to find relevant experts for given input queries.
The following steps are required for Setup:  
1. **Download required model files:**  
Download [ensemble_model.model](https://drive.google.com/file/d/1BhhIocduWQI-RKqhmqRhDo7bJUK7tVoS/view) to data/models  
Download [expert_voting.model](https://drive.google.com/file/d/1pM6kUf7xxTkSJ7MsDqkVb07hUYrNsZB9/view) to data/models  
Download [bm25_oqe.model]() to data/models/tfidf  
Download [sent2vec_oqe.model]() to data/models/sent2vec  
Download [wiki_bigrams.bin](https://drive.google.com/file/d/0B6VhzidiLvjSaER5YkJUdWdPWU0/view) to data/models/sent2vec (warning, this file has a size of 17gb the smaller unigram sent2vec model can also be used: https://github.com/epfml/sent2vec). 

2. **Change directory to paper_retrieval/server**
```
$ cd paper_retrieval/server
```

3. **Install python requirements (need python 3.7)**
```
$ pip install -r requirements.txt
```

4. **Run flask server**
```
$ python flask_server.py
```

The server can now be accessed under http://localhost:8071/experts?query=blockchain&limit=20

### Repository structure
data: contains corpus file with all paper abstracts, evaluation results as csv and model files

data_collector: contains code for collection all data from the Microsoft Academic Graph and saving it to a MongoDB. 
Form there the data is queried and preprocessed into a single document corpus.

paper_retrieval: contains the code for running the expert retrieval. This includes all training/evaluation jupyter notebooks

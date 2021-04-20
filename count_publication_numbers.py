import json
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import MultiLabelBinarizer
from utils import count_word
from datetime import datetime
from collections import defaultdict


online_client = MongoClient("mongodb+srv://Olivier:Olivier7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority")
db_online = online_client.get_database("GLO-7027-13")

collection = db_online.get_collection("test_publication_info")

test = collection.find().next()

articles_wordcount = {}
i=0

publication = defaultdict(set)

for article in collection.find({'publications.publicationDate': {'$gte': datetime(2019, 1, 1)}}):
    publication[article['id']] |= {article['organizationKey']}

for key, value in publication.items():
    publication[key] = list(value)

nom_journaux = set()
for value in publication.values():
    for journal in value:
        nom_journaux.add(journal)

df = pd.json_normalize(publication).transpose()

mlb = MultiLabelBinarizer()
title = pd.DataFrame(mlb.fit_transform(df[0]),columns=mlb.classes_, index=df.index)

title.to_csv('publication_count_test.csv')

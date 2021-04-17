import json
from pymongo import MongoClient
from utils import count_word
from datetime import datetime


online_client = MongoClient("mongodb+srv://Olivier:Olivier7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority")
db_online = online_client.get_database("GLO-7027-13")

collection = db_online.get_collection("train_article_info")

test = collection.find().next()

articles_wordcount = {}
i=0

for article in collection.find({'type': 'article', 'creationDate': {'$gte': datetime(2019, 1, 1)}}):
    i+=1
    print(i)
    wordcount = 0
    title_wordcount = 0
    for chapter in article['title'].values():
        title_wordcount += count_word(chapter)
    for chapter in article['chapters']:
        if chapter['type'] == 'paragraph':
            text = [text for text in chapter['text'].values()]
            for t in text:
                wordcount += count_word(t)
    articles_wordcount[article['id']] = wordcount

with open("Articles_wordcount.json", 'w') as f:
    json.dump(articles_wordcount, f)
    f.close()
with open("title_wordcount.json", 'w') as f:
    json.dump(articles_wordcount, f)
    f.close()
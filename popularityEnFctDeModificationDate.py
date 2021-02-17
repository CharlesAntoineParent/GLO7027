from utils import *
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime



with open('articleByAuthor.pickle', 'rb') as handle:
    data = pickle.load(handle)



sample_size = 1000


sampleDict = dict()

for journal in data.keys():
    articleList = list(data[journal].keys())
    random.shuffle(articleList)
    for article in articleList[0:sample_size]:
        try:
            articleInfo = getArticle(article)
            articleModDate = articleInfo[1]['modificationDate']
            articleCreationDate = articleInfo[1]['creationDate']
            sampleDict[article] = data[journal][article]
            sampleDict[article]['modificationDate'] = articleModDate
            sampleDict[article]['creationDate'] = articleCreationDate
        except FileNotFoundError:
            print(article)
            continue


timeDifferentialModificationCreation = dict()
for article in sampleDict.keys():
    print(article)
    creationDate = sampleDict[article]['creationDate'].replace('T',' ').replace('Z','')
    creationDate = datetime.datetime.strptime(creationDate, '%Y-%m-%d %H:%M:%S.%f')
    modificationDate = sampleDict[article]['modificationDate'].replace('T',' ').replace('Z','')
    modificationDate = datetime.datetime.strptime(modificationDate, '%Y-%m-%d %H:%M:%S.%f')
    timeDifferentialModificationCreation[article] = ((modificationDate - creationDate).total_seconds())/3600


plt.hist(timeDifferentialModificationCreation.values(),50,range = (0,100))
plt.title('Différence entre la date de modification et de création')
plt.ylabel("Nombre d'heures")
plt.show()
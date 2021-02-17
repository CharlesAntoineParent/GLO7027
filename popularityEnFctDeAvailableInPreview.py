from utils import *
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt



with open('articleByAuthor.pickle', 'rb') as handle:
    data = pickle.load(handle)



sample_size = 10000


sampleDict = dict()

for journal in data.keys():
    articleList = list(data[journal].keys())
    random.shuffle(articleList)
    for article in articleList[0:sample_size]:
        try:
            articleInfo = getArticle(article)
            articleType = articleInfo[1]['availableInPreview']
            sampleDict[article] = data[journal][article]
            sampleDict[article]['availableInPreview'] = articleType
        except FileNotFoundError:
            print(article)
            continue


df = pd.DataFrame.from_dict(sampleDict, orient='index')

df.groupby('availableInPreview').sum()

## Only False
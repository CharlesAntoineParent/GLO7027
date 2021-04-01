from utils import *
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load('fr_core_news_sm')
import operator
import random 


with open('data/article2019.pickle', 'rb') as handle:
    data = pickle.load(handle)


sampleSize = 2000


a = dict()
nbArticle = 0
for journal in data.keys():
    articleList = list(data[journal].keys())
    random.shuffle(articleList)
    articleList = articleList[0:sampleSize]
    for article in articleList:
        try:
            articleInfo = getArticle(article)
            nbArticle += 1
            typeArticle = dict()
            for i in articleInfo[1]['chapters']:
                typeArticle[i['type']] = 1

            for i,j in typeArticle.items():
                try:
                    a[i] += 1
                except KeyError:

                    a[i] = 1
            try:
                typeArticle['paragraph']
            except KeyError:
                print(articleInfo[1]['title'],article)
               

        except (FileNotFoundError,KeyError):
            pass


plt.bar(a.keys(),a.values())
plt.axhline(y=12000,linewidth=1.5, color='r',label="nombre d'articles échantillon")
plt.ylabel("Nombre d'articles contenant au moins un chapter de ce type")
plt.xlabel("Type de Chapter")
plt.ylim(0,14000)
plt.legend()
plt.show()




sampleSize = 2000


a = dict()
nbArticle = 0
for journal in data.keys():
    articleList = list(data[journal].keys())
    random.shuffle(articleList)
    articleList = articleList[0:sampleSize]
    for article in articleList:
        try:
            articleInfo = getArticle(article)
            
            a[article] = list()
            for i in articleInfo[1]['chapters']:
                a[article].append(i['type'])


        except (FileNotFoundError,KeyError):
            pass


paragraph = [i.count('paragraph') for i in list(a.values())]

plt.boxplot([paragraph])
plt.xticks([1], ['paragraphe'])
plt.ylabel('Quantité par article')
plt.grid()
plt.show()

photo = [i.count('photo') for i in list(a.values())]
photoPosition = [i.index('photo') for i in list(a.values()) if 'photo' in i]
photoPosition

video = [i.count('video') for i in list(a.values())]
videoPosition = [i.index('video') for i in list(a.values()) if 'video' in i]

plt.boxplot([photoPosition,videoPosition])
plt.xticks([1,2], ['photo','video'])
plt.ylabel("Position du contenu dans l'article")
plt.grid()
plt.show()


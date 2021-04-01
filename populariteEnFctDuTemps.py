from utils import *
import pandas as pd
import datetime
from tqdm import tqdm
from time import sleep
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np



with open('data/article2019.pickle', 'rb') as handle:
    data = pickle.load(handle)

for article in list(data['lesoleil'].keys())[0:1000]:
    try:
        print(data['lesoleil'][article]['author'])
    except KeyError:
        pass


startDate = datetime.date(2019,1,1)
endDate = datetime.date(2019,7,31)
dateList = [startDate + datetime.timedelta(days=day) for day in range( (endDate - startDate).days + 1 )]

journalPopularityByday = dict()

Hash2019 = set([i for i in data['lesoleil'].keys()])

ScoreParJourParAricle = dict()

for day in dateList:
    print(day)
    df = dayInfoSummary(str(day),'lesoleil')
    for article in df.keys():
        if ( article in Hash2019):
            Score = df[article]['View'] + df[article]['View5'] + 2 *df[article]['View10'] +  5 * df[article]['View30'] + 10 *df[article]['View60']
            try:
                ScoreParJourParAricle[article].append(Score)
            
            except KeyError:
                ScoreParJourParAricle[article] = list()
                ScoreParJourParAricle[article].append(Score)



for j,i in list(ScoreParJourParAricle.items()):
    if 90 < len(i):
        continue
    else:
        del ScoreParJourParAricle[j]

for j,i in list(ScoreParJourParAricle.items()):
    newList = [views/np.sum(i) for views in i]
    ScoreParJourParAricle[j] = newList


df1 = pd.DataFrame.from_dict(ScoreParJourParAricle,orient='index')
yRepartition = list()
xRepartition = list()
for i,j in df1.items():
    xRepartition.append(i)
    yRepartition.append(np.mean(j))


z = np.polyfit(xRepartition[0:90], yRepartition[0:90], 4)
p = np.poly1d(z)
#plt.plot(xRepartition[0:90],p(xRepartition[0:90]),"r--")
plt.plot(xRepartition[0:30],yRepartition[0:30])
plt.xlabel("Jours écoulés depuis la publication")
plt.ylabel("Densité des scores")
plt.title("Fonction de densité des scores par jour écoulés depuis la publication de l'article")
plt.grid()
plt.show()


ArticleScore = dict()
for journal in list(data.keys()):
    for article in data[journal].keys():
        try:
           ArticleScore[article] += data[journal][article]['View'] + data[journal][article]['View5'] + 2 *data[journal][article]['View10'] +  5 * data[journal][article]['View30'] + 10 *data[journal][article]['View60'] 
        except KeyError:
            ArticleScore[article] = data[journal][article]['View'] + data[journal][article]['View5'] + 2 *data[journal][article]['View10'] +  5 * data[journal][article]['View30'] + 10 *data[journal][article]['View60']


score2019 = [np.sum(i) for i in ScoreParJourParAricle.values()]
score2019.sort()
score2019cut = score2019[round(0.1*len(score2019)):round(0.9*len(score2019))]
scoreDict = {"Ensemble des scores de 2019":score2019}
fig, ax = plt.subplots()
ax.boxplot(scoreDict.values(),vert=False)
plt.yticks([1],['Scores de 2019'])
plt.title('boîte à moustache du score des articles de 2019')
plt.xlabel('Score')
plt.grid()
ax.ticklabel_format(axis='x',style='plain')
plt.show()


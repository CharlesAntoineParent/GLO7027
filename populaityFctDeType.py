from utils import *
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt



with open('articleByAuthor.pickle', 'rb') as handle:
    data = pickle.load(handle)


sample_size = 4000


sampleDict = dict()

for journal in data.keys():
    articleList = list(data[journal].keys())
    random.shuffle(articleList)
    for article in articleList[0:sample_size]:
        try:
            articleInfo = getArticle(article)
            articleType = articleInfo[1]['type']
            sampleDict[article] = data[journal][article]
            sampleDict[article]['type'] = articleType
        except FileNotFoundError:
            print(article)
            continue




df = pd.DataFrame.from_dict(sampleDict, orient='index')


DossierDf = df[df['type'] == 'dossier']
nbOfView = list(DossierDf.groupby('type').sum().iloc[0].values)
viewLabel = list(DossierDf.groupby('type').sum().iloc[0].index)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(nbOfView, labels = viewLabel,autopct='%1.2f%%')
plt.show()

DossierDf['Score'] = DossierDf['View'] + DossierDf['View5'] + 2 *DossierDf['View10'] +  5 * DossierDf['View30'] + 10 *DossierDf['View60']



ArticleDf = df[df['type'] == 'article']

nbOfView = list(ArticleDf.groupby('type').sum().iloc[0].values)
viewLabel = list(ArticleDf.groupby('type').sum().iloc[0].index)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(nbOfView, labels = viewLabel,autopct='%1.2f%%')
plt.show()

ArticleDf['Score'] = ArticleDf['View'] + ArticleDf['View5'] + 2 *ArticleDf['View10'] +  5 * ArticleDf['View30'] + 10 *ArticleDf['View60']

dataBoxPlot = [list(ArticleDf['Score']),list(DossierDf['Score'])]

fig = plt.figure(figsize =(10, 7)) 
ax = fig.add_subplot(111)

bp = ax.boxplot(dataBoxPlot) 
ax.set_xticklabels(['Score Article','Score Dossier'])
plt.title("Box plot score article et dossier")
plt.show() 

## Répartition très similaire




getArticle('f361c7595ff5a56684ed3a3fbf56b420')

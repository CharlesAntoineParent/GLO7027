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

list(sampleDict.values())[0]['date'][0]

sampleDict = dict()
sampleSize = 15000

for journal in data.keys():
    articleList = list(data[journal].keys())
    random.shuffle(articleList)
    articleList = articleList[0:sampleSize]
    for article in articleList:
        try:
            articleInfo = getArticle(article)
            articleModDate = articleInfo[1]['title']
            
            sampleDict[article] = data[journal][article]
            date = articleInfo[0][0]['publications'][0]['publicationDate'].split('T')[0]
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            sampleDict[article]['title'] = articleModDate
            sampleDict[article]['date'] = date
        except FileNotFoundError:
            print(article)
            continue


spacy.lang.fr.stop_words.STOP_WORDS

TitleTest = list(sampleDict.values())[0]['title']['fr']

introduction_doc = nlp('«Détresse psychologique» pour les 10 enfants évacués en Syrie')

def customTitleTokenizer(title):
    
    token_list = list()
    spacySentence = nlp(title.lower())

    for token in spacySentence:
        
        if (token.text not in spacy.lang.fr.stop_words.STOP_WORDS):
            
                
                if not token.is_punct:
                    if not token.text.isdigit():
                        if (token.lemma_ not in spacy.lang.fr.stop_words.STOP_WORDS):
                            if (token.lemma_.find("\\") == -1):
                                token_list.append((token.lemma_))

    return token_list


for article in sampleDict.keys():
    sampleDict[article]['score'] = sampleDict[article]['View'] + sampleDict[article]['View5'] + 2 *sampleDict[article]['View10'] +  5 * sampleDict[article]['View30'] + 10 *sampleDict[article]['View60']



for article in sampleDict.keys():

    sampleDict[article]['tokenizedTitle'] = customTitleTokenizer(sampleDict[article]['title']['fr'])



wordFrequence = dict()
for article in sampleDict.keys():

    for word in sampleDict[article]['tokenizedTitle']:
        try:
            wordFrequence[word] += 1

        except KeyError:
            wordFrequence[word] = 1



sortedWordFrequence = sorted(wordFrequence.items(), key=operator.itemgetter(1))
sortedWordFrequence15AndMore = [i for i in sortedWordFrequence if i[1] > 14]
sortedWordFrequence15AndMore.reverse()

wordVector = dict()
for i in sortedWordFrequence:
    wordVector[i[0]] = list()
    for article in sampleDict.keys():
        if (i[0] in sampleDict[article]['tokenizedTitle']):
            wordVector[i[0]].append(1)
        else:
            wordVector[i[0]].append(0)


df = pd.DataFrame.from_dict(wordVector, orient='index')

freqByMonth = dict()
for word in sortedWordFrequence15AndMore[0:200]:
    freqByMonth[word[0]] = dict()
    for month in range(1,13):
        freqByMonth[word[0]][month] = 0


for article in sampleDict.values():
    month = article['date'].month
    for word in sortedWordFrequence15AndMore[0:200]:
        
        if word[0] in article["tokenizedTitle"]:
            freqByMonth[word[0]][month] += 1


for article in sampleDict.values():
    if 'projet' in article["tokenizedTitle"]:
        print(article['title'])
        print(article['date'])



df = pd.DataFrame.from_dict(freqByMonth,orient='index')

festival = df.loc['festival'][0:7]
maison = df.loc['maison'][0:7]
granby = df.loc['granby'][0:7]
mois = ['Janvier','Février','Mars','Avril','Mai',"Juin",'Juillet']
plt.plot(mois,maison, label = "maison")
plt.plot(mois,granby, label = "granby")
plt.plot(mois,festival, label = "festival")
plt.title("Fréquence des mots dans des titres par mois")
plt.ylabel("Fréquence du mot dans le titre")
plt.xlabel("Mois")
plt.ylim(0,100)
plt.legend()
plt.grid()
plt.show()


sortedWordFrequence.reverse()

def getWordMeanScore(word, sampleDict):
    
    listScore = list()
    for article in sampleDict.values():
        
        if word in article['tokenizedTitle']:
            
            print(article['title'])
            
            listScore.append(article['score'])

    return listScore


b=getWordMeanScore(sortedWordFrequence[8][0],sampleDict)

sortedWordFrequence[0:58]

fig = plt.figure(figsize =(10, 7)) 
ax = fig.add_subplot(111)
ax.boxplot([getWordMeanScore(sortedWordFrequence[i][0],sampleDict) for i in range(0,20)])
ax.set_xticklabels([sortedWordFrequence[i][0] for i in range(0,20)])
plt.show()


sortedWordFrequence.reverse()
plt.bar([sortedWordFrequence[i][0] for i in range(0,10)], [sortedWordFrequence[i][1] for i in range(0,10)])
plt.xlabel("mots")
plt.ylabel("Nombre de fois utilisés")
plt.show()



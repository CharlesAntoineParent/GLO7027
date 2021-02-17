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



with open('articleByAuthor.pickle', 'rb') as handle:
    data = pickle.load(handle)



sample_size = 3000


sampleDict = dict()


for journal in data.keys():
    articleList = list(data[journal].keys())
    random.shuffle(articleList)
    for article in articleList[0:sample_size]:
        try:
            articleInfo = getArticle(article)
            articleModDate = articleInfo[1]['title']
            
            sampleDict[article] = data[journal][article]
            sampleDict[article]['title'] = articleModDate
            
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



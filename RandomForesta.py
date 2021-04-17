from collections_utilis import * 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from scipy.stats import ttest_ind
import scipy
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.tree import export_graphviz
import pydot
import json

def mann_whitney_u_test(distribution_1, distribution_2):
    u_statistic, p_value = scipy.stats.mannwhitneyu(distribution_1, distribution_2)
    return u_statistic, p_value


#### MAIN FUNCTION ####
#Perform the Mann-Whitney U Test on the two distributions



df = pd.read_pickle('test.pkl')


wordDict = dict()
for i in df['title'].values:
    try:
        for word in i:     
            try:
                wordDict[word] += 1
            except KeyError:
                wordDict[word] = 1
    except TypeError:
        pass

for i,j in wordDict.copy().items():
    if j < 400:
        del wordDict[i]


def wordSignifiance(word):
    ScoreAvecMot = list()
    ScoreSansMot = list()
    for i,j in zip(df['title'].values, df['score'].values):
        try:
            if word in i:
                ScoreAvecMot.append(j)
            else:
                ScoreSansMot.append(j)
        except TypeError:
            ScoreSansMot.append(j)

    return ttest_ind(ScoreAvecMot, ScoreSansMot)



df['creationDate']
dummiesJourSemaine = pd.get_dummies(df['creationDate'])
dummiesJourSemaine.columns = ['lundi','mardi','mercredi','jeudi','vendredi','samedi','dimanche']

df = pd.concat([df,dummiesJourSemaine],axis=1)

paragraph = list()
video = list()
photo =list()
html = list()
slideshow = list()
quote = list()
for i in df.chapters.values:
    try:
        paragraph.append(i.count('paragraph'))
        video.append(i.count('video'))
        photo.append(i.count('photo'))
        html.append(i.count('html'))
        slideshow.append(i.count('slideshow'))
        quote.append(i.count('quote'))
    except (IndexError,TypeError, AttributeError):
        paragraph.append(0)
        video.append(0)
        photo.append(0)
        html.append(0)
        slideshow.append(0)
        quote.append(0)


df['paragraph'] = paragraph
df["video"] = video
df["photo"] = photo
df["html"] = html
df["slideshow"] = slideshow
df["quote"] = quote

listauteur = dict()

for i in df.authors:
    try:
        for j in i:
            try:
                listauteur[j] += 1
            except KeyError:
                listauteur[j] = 1
    except TypeError:
        pass
listauteurSplitByPopularity = listauteur.copy()
for i,j in listauteur.copy().items():
    if j < 500:
        del listauteur[i]

for auteur in listauteur.keys():
    df[auteur] = [0] * len(df.index)

def populariteAuteur(x,mean,std,dictPop):
    
    try:
        return np.array([(dictPop[i]-mean)/std for i in x]).mean()
    except TypeError:

        return 0

def countAuteur(x, auteur):
    try:
        return x.count(auteur)
    except AttributeError:
        return 0
for auteur in listauteur.keys():
    df[auteur] = df['authors'].apply(lambda x: countAuteur(x,auteur))

def countWord(x, word):
    try:
        return x.count(word)
    except AttributeError:
        return 0


df['populariteAuteur'] = df["authors"].apply(lambda x: populariteAuteur(x,np.array(list(listauteurSplitByPopularity.values())).mean(),np.array(list(listauteurSplitByPopularity.values())).std(),listauteurSplitByPopularity))

dummiesOrganisation = pd.get_dummies(df['organizationKey'])
df = pd.concat([df,dummiesOrganisation],axis=1)
df['populariteAuteur'] = df['populariteAuteur'].fillna(0)



dummiesPublication = pd.get_dummies(df['publications'])
df = pd.concat([df,dummiesPublication],axis=1)
df.externalIds = df.externalIds.fillna(0)
df['score'] = df['point_view5'].fillna(0) + df['point_view10'].fillna(0) + df['point_view30'].fillna(0) + df['point_view60'].fillna(0)
wordToKeep = list()
for word in wordDict.keys():
    if wordSignifiance(word)[1] < 0.01:
        wordToKeep.append(word)

for word in wordToKeep:
    df[f'word_{word}'] = [0] * len(df.index)

for word in wordToKeep:
    df[f'word_{word}'] = df[f'word_{word}'].apply(lambda x: countWord(x,auteur))

title_word_count = pd.DataFrame.from_dict(json.load(open('title_wordcount.json')), orient='index')
title_word_count.columns = ['title_word_count']
article_word_count = pd.DataFrame.from_dict(json.load(open('Articles_wordcount.json')), orient='index')
article_word_count.columns = ['article_word_count']

df = pd.merge(df, title_word_count, left_on='_id', right_index=True, how='left')
df = pd.merge(df, article_word_count, left_on='_id', right_index=True, how='left')

# twitter_username = pd.DataFrame.from_dict(json.load(open('twitter_author_name.json')), orient='index')
# twitter_username.columns = ['twitter']
# twitter_followers = pd.DataFrame.from_dict(json.load(open('twitter_followers.json')), orient='index')
# twitter_followers.columns = ['twitter_followers']
# df = pd.merge(df, twitter_username, left_on='_id', right_index=True, how='left')
# df = pd.merge(df, twitter_followers, left_on='twitter', right_index=True, how='left')
# df = df.drop(['twitter'])
# df['twitter_followers'] = df['twitter_followers'].fillna(0)
# df['twitter_followers'] = df['twitter_followers'].replace(np.nan, 0)

df = df.drop(['_id','creationDate','authors','title','channel','chapters','organizationKey','hash','count','score','source'],axis=1)
df['score'] = df['point_view5'].fillna(0) + df['point_view10'].fillna(0) + df['point_view30'].fillna(0) + df['point_view60'].fillna(0)
df = df.drop(['view','view5','view10','view30','view60','point_view','point_view5','point_view10','point_view30','point_view60'],axis=1)

newDf = pd.DataFrame()
for slug in df['publications'].value_counts().keys():
    df_temp = df[df['publications'] == slug]
    df_temp['score'] = list(pd.qcut(df_temp['score'].rank(method='first'), q=20, labels=list(range(1,21))).values)
    newDf = pd.concat([newDf,df_temp],axis=0)

df = newDf
df = df.drop(['publications'],axis=1)

df = df.fillna(0)

X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def evalWorst(X_test,Y,model):
    category = ['actualites','affaires', 'arts', 'auto', 'cahiers-speciaux', 'chroniques', 'cinema',
                'contenu-commandite', 'la-vitrine', 'le-mag', 'maison', 'monde',
                'opinions', 'science', 'sports']
    Y_test = Y
    predictions = model.predict(X_test)
    dictByChannel = dict()
    for i in range(len(X_test)):
        
        categoryTmp = category[list(X_test[i][77:92]).index(1)]
        
        try:
            dictByChannel[categoryTmp].append((predictions[i],Y_test[i]))
        except KeyError:
            dictByChannel[categoryTmp] = [(predictions[i],Y_test[i])]

    
    intersection = 0
    union = 0
    for i in dictByChannel.keys():
        quantile10Pred = np.quantile(np.array([i[0] for i in dictByChannel[i]]),0.1)
        quantile10True = np.quantile(np.array([i[1] for i in dictByChannel[i]]),0.1)
        
        for j in dictByChannel[i]:
            if (quantile10Pred >= j[0] or  quantile10True >= j[1]) :
                union += 1
                if (quantile10Pred >= j[0] and  quantile10True >= j[1]) :
                    intersection += 1

    
    return intersection/union


n_features = X_train.shape[1]
n_estimators = [300,500,100]
max_features = ['auto']
min_samples_split = [18, 20,22,24]
bootstrap = [True]

grid_search = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'bootstrap': bootstrap}
rf = RandomForestRegressor()
rf_grid_search = GridSearchCV(estimator = rf, param_grid = grid_search, 
                          cv = 3, n_jobs = -1, verbose = 2)
rf_grid_search.fit(X_train, y_train)
rf_grid_search
rf_grid_search.best_params_
predictions = rf_grid_search.predict(X_test)
evalWorst(X_train, y_train,rf_grid_search.best_estimator_)

bestparamsRf = {'bootstrap': True, 'max_features': 'auto', 'min_samples_split': 20, 'n_estimators': 500}



n_features = X_train.shape[1]
n_estimators = [1024,1500,2056]
max_depth = [8,10,16]
learning_rate = [0.1,0.15,0.20,0.25]
max_features = ['auto']
min_samples_split = [10,15,20,7]
subsample = [1]
grid_search_GB = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'subsample': subsample,
               'max_features': max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split}

clf = GridSearchCV(GradientBoostingRegressor(), grid_search_GB, cv=3, n_jobs=-1,verbose = 2)
clf.fit(X_train, y_train)
clf.best_params_
evalWorst(X_train, y_train,clf.best_estimator_)
bestparamsGb = {'learning_rate': 0.15, 'max_depth': 8, 'max_features': 'auto', 'min_samples_split': 15, 'n_estimators': 1024, 'subsample': 1}


RandomForest = RandomForestRegressor(bootstrap= True, max_features='auto', min_samples_split=20, n_estimators=500)
GradiantBoosting = GradientBoostingRegressor(learning_rate=0.15, max_depth=8, max_features='auto',min_samples_split=15,n_estimators=1024,subsample=1)
finalModel = VotingRegressor([('RandomForest', RandomForest), ('GradiantBoosting', GradiantBoosting)])
finalModel.fit(X_train, y_train)
evalWorst(X_train, y_train,finalModel)
evalWorst(X_train, y_train,GradiantBoosting)
evalWorst(X_test, y_test,GradiantBoosting)
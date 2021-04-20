import decimal
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_selection import RFECV
from collections_utilis import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import ttest_ind
import scipy
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.tree import export_graphviz
import pydot
from joblib import load, dump
import json
import FeaturesEvaluator
import pickle as pkl


def mann_whitney_u_test(distribution_1, distribution_2):
    u_statistic, p_value = scipy.stats.mannwhitneyu(distribution_1, distribution_2)
    return u_statistic, p_value


#### MAIN FUNCTION ####
# Perform the Mann-Whitney U Test on the two distributions


df = pd.read_pickle('test_data.pkl')

wordDict = json.load(open('word_dict.json', 'r'))

def wordSignifiance(word):
    ScoreAvecMot = list()
    ScoreSansMot = list()
    for i, j in zip(df['title'].values, df['score'].values):
        try:
            if word in i:
                ScoreAvecMot.append(j)
            else:
                ScoreSansMot.append(j)
        except TypeError:
            ScoreSansMot.append(j)

    return ttest_ind(ScoreAvecMot, ScoreSansMot)

dummiesJourSemaine = pd.get_dummies(df['creationDate'])
dummiesJourSemaine.columns = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']

df = pd.concat([df, dummiesJourSemaine], axis=1)

paragraph = list()
video = list()
photo = list()
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
    except (IndexError, TypeError, AttributeError):
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

listauteur = json.load(open('liste_auteur.json', 'r'))

listauteurSplitByPopularity = listauteur.copy()
for i, j in listauteur.copy().items():
    if j < 500:
        del listauteur[i]

for auteur in listauteur.keys():
    df[auteur] = [0] * len(df.index)


def populariteAuteur(x, mean, std, dictPop):
    try:
        return np.array([(dictPop[i] - mean) / std for i in x]).mean()
    except TypeError:
        return 0
    except KeyError:
        return 0


def countAuteur(x, auteur):
    try:
        return x.count(auteur)
    except AttributeError:
        return 0


for auteur in listauteur.keys():
    df[auteur] = df['authors'].apply(lambda x: countAuteur(x, auteur))


def countWord(x, word):
    try:
        return x.count(word)
    except AttributeError:
        return 0


df['populariteAuteur'] = df["authors"].apply(
    lambda x: populariteAuteur(x, np.array(list(listauteurSplitByPopularity.values())).mean(),
                               np.array(list(listauteurSplitByPopularity.values())).std(), listauteurSplitByPopularity))

dummiesOrganisation = pd.get_dummies(df['organizationKey'])
df = pd.concat([df, dummiesOrganisation], axis=1)
df['populariteAuteur'] = df['populariteAuteur'].fillna(0)

dummiesPublication = pd.get_dummies(df['publications'])
df = pd.concat([df, dummiesPublication], axis=1)
df.externalIds = df.externalIds.fillna(0)


title_word_count = pd.DataFrame.from_dict(json.load(open('title_wordcount_test.json')), orient='index')
title_word_count.columns = ['title_word_count']
article_word_count = pd.DataFrame.from_dict(json.load(open('Articles_wordcount_test.json')), orient='index')
article_word_count.columns = ['article_word_count']

df = pd.merge(df, title_word_count, left_on='_id', right_index=True, how='left')
df = pd.merge(df, article_word_count, left_on='_id', right_index=True, how='left')

twitter_username = pd.DataFrame.from_dict(json.load(open('twitter_author_name.json')), orient='index')
twitter_username.columns = ['twitter']
twitter_followers = pd.DataFrame.from_dict(json.load(open('twitter_followers.json')), orient='index')
twitter_followers.columns = ['twitter_followers']
df = pd.merge(df, twitter_username, left_on='_id', right_index=True, how='left')
df = pd.merge(df, twitter_followers, left_on='twitter', right_index=True, how='left')
df = df.drop(['twitter'], axis=1)
df['twitter_followers'] = df['twitter_followers'].fillna(0)
df['twitter_followers'] = df['twitter_followers'].replace(np.nan, 0)

publications = pd.read_csv('publication_count_test.csv', index_col=0)

df = pd.merge(df, publications, left_on='_id', right_index=True, how='left')

df = df.drop(
    ['creationDate', 'authors', 'title', 'channel', 'chapters', 'organizationKey'],
    axis=1)

newDf = pd.DataFrame()
for slug in df['publications'].value_counts().keys():
    df_temp = df[df['publications'] == slug]
    newDf = pd.concat([newDf, df_temp], axis=0)

df = newDf




df = df.drop(['publications'], axis=1)

df = df.fillna(0)

X = df.iloc[:, :]


id = X._id


X = X.drop("_id", axis = 1)

good_index = ['externalIds', 'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi',
       'samedi', 'dimanche', 'paragraph', 'video', 'photo', 'html', 'quote',
       'louis-denis ebacher', '', 'justine mercier', 'sylvain st-laurent',
       'michel tasse', 'karine blanchard', 'afp', 'claude plante',
       'marc rochette', 'olivier bosse', 'marie-eve martel',
       'richard therrien', 'julien paquette', 'carrefour des lecteurs',
       'mylene moisan', 'carl tardif', 'josianne desloges', 'isabelle mathieu',
       'marc allard', 'stephane begin', 'marie-eve lambert', 'isabelle legare',
       'patrick duquette', 'rene-charles quirion', 'marie-eve lafontaine',
       'françois bourque', 'anne-marie gravel', 'daniel leblanc',
       'denis gratton', 'populariteAuteur', 'latribune_x', 'lavoixdelest_x',
       'ledroit_x', 'lenouvelliste_x', 'lequotidien_x', 'lesoleil_x',
       'actualites', 'affaires', 'arts', 'chroniques', 'la-vitrine', 'le-mag',
       'maison', 'monde', 'opinions', 'sports', 'title_word_count',
       'article_word_count', 'latribune_y', 'lavoixdelest_y', 'ledroit_y',
       'lenouvelliste_y', 'lequotidien_y', 'lesoleil_y']

X_clean = X.loc[:, good_index]


def predictWorst(X, ids, model, id_cat):

    predictions = model.predict(X)
    dictByChannel = defaultdict(list)

    for i in range(len(X)):
        categoryTmp = id_cat[ids[i]]

        dictByChannel[categoryTmp].append((ids[i], predictions[i]))

    nb_article_per_cat = Counter(id_cat.values())


    nb_pred = {}

    for key, value in nb_article_per_cat.items():
        nb_pred[key] = int(decimal.Decimal(value/10).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

    for key, value in dictByChannel.items():
        value.sort(key=lambda x: x[1])
        dictByChannel[key] = value[:nb_pred[key]]



    return dictByChannel

categories_prof = pd.read_csv(open('listecatégories.txt'), sep='\t', header=None)
categories = dict(categories_prof.to_numpy())

np.unique(categories_prof[1])

gb = pkl.load(open('best_classifier.p', 'rb'))

worst_articles = predictWorst(X_clean, id, gb, categories)

all_worst_articles = []

for value in worst_articles.values():
    all_worst_articles += value

all_worst_articles = [x[0] for x in all_worst_articles]
pd.DataFrame(all_worst_articles).to_csv('predicted_articles.csv', index=None)
import configparser
import json
import os
import operator
import pandas as pd
from datetime import datetime, date
import dateutil.parser
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections_utilis import *
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load('fr_core_news_sm')
import operator
import random



sns.set_theme(style="ticks", color_codes=True)

tips = sns.load_dataset("tips")


configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

seasons = {'winter': (date(2019,  1,  1),  date(2019,  3, 20)),
           'spring': (date(2019,  3, 21),  date(2019,  6, 20)),
           'summer': (date(2019,  6, 21),  date(2019,  9, 22)),
           'autumn': (date(2019,  9, 23),  date(2019, 12, 20)),
           'winter': (date(2019, 12, 21),  date(2019, 12, 31))}

score_by_view_type = {'View': 1,
                      'View5': 1,
                      'View10': 2,
                      'View30': 5,
                      'View60': 10}

all_organisations = ["latribune",
                     "lavoixdelest",
                     "ledroit",
                     "lenouvelliste",
                     "lequotidien",
                     "lesoleil"]


def getArticleByPath(path):
    ArticleInfo = list()

    for file in path:
        try:
            with open(file, 'r') as article:
                ArticleInfo.append(json.load(article))
        except FileNotFoundError:
            file = file.replace('train','test')
            with open(file, 'r') as article:
                ArticleInfo.append(json.load(article))


    return ArticleInfo


def getArticle(p_hash):
    fileInfoPath = f'{dataPath}/train/{p_hash}'
    selectedJsonFiles = [f'{fileInfoPath}--publication-info.json',f'{fileInfoPath}.json']

    ArticleInfo = getArticleByPath(selectedJsonFiles)

    return ArticleInfo


def getDayInfo(p_date,p_journal):

    journalPath = f'{dataPath}/analytics/{p_journal}/'

    concernedFile = [f'{journalPath}{file}' for file in os.listdir(journalPath) if file.startswith(p_date)]

    dayInfo = list()

    for day in concernedFile:
        with open(day,'r') as file:
            dayInfo = json.load(file)

    return dayInfo




def dayInfoSummary(p_date,p_journal):

    dayInfo = getDayInfo(p_date,p_journal)

    dayInfoSummary = dict()
    for view in dayInfo :
        try:
            dayInfoSummary[view['hash']][view['name']] += 1
        except KeyError:
            dayInfoSummary[view['hash']] = {'View':0,'View5':0,'View10':0,'View30':0,'View60':0}
            dayInfoSummary[view['hash']][view['name']] += 1


    return dayInfoSummary


def get_all_hash(p_date):
    set_of_hash = set()
    organization_keys = ["ledroit", "lesoleil", "lenouvelliste", "lequotidien", "latribune", "lavoixdelest"]
    for organization_key in organization_keys:
        info_of_the_day_for_organisation = dayInfoSummary(p_date, organization_key)
        for hash_article in info_of_the_day_for_organisation.keys():
            set_of_hash.add(hash_article)

    return list(set_of_hash)




def dayPopularity(p_date,p_journal):

    p_dayInfoSummary = dayInfoSummary(p_date,p_journal)

    dayPopularity = dict()
    for articleHash, views in p_dayInfoSummary.items():
        dayPopularity[articleHash] = (views['View']) + (views['View5']) + (views['View10'] * 2) + (views['View30'] * 5) + (views['View60'] * 10)

    return dayPopularity

def get_score_from_views(views):
    score = 0
    for view_type, count in views.items():
        score += score_by_view_type[view_type] * count
    return score


def get_views_by_hash(hash, organisations = all_organisations):
    views = {'View':0,
             'View5':0,
             'View10':0,
             'View30':0,
             'View60':0}
    for organisation in organisations:
        analyticPath = f'{dataPath}/analytics/{organisation}'
        for filename in os.listdir(analyticPath):
            analytics = json.load(filename)
            for view in analytics:
                if view["hash"] == hash:
                    views[view['name']]+=1


def get_all_scores_and_view_per_organization_for_a_day(p_date):
    organization_keys = ["ledroit", "lesoleil", "lenouvelliste", "lequotidien", "latribune", "lavoixdelest"]
    all_hashes_of_the_day = get_all_hash(p_date)
    dict_of_score_and_views_for_the_day = {}

    for article_hash in all_hashes_of_the_day:
        dict_of_score_and_views_for_the_day[article_hash] = np.zeros((len(organization_keys) + 1, 7), dtype = np.uint)

    #dict_of_score_and_views_for_the_day = { your_key: dict_of_score_and_views_for_the_day[your_key] for your_key in ['ff2d4b6b86b58f9cdc361eb5553b0480', '2f930a2b431e14ec595cb42274f1317b', '1d4aa70686ce4e4e0e47f6037347efa2'] }

    for index, organization_key in enumerate(organization_keys):
        info_for_organization = dayInfoSummary(p_date, organization_key)
        #info_for_organization = dict(list(info_for_organization.items())[0: 3])

        for article_hash, views in info_for_organization.items():

            total_score = get_score_from_views(views)
            list_of_score = [1, views['View'], views['View5'], views['View10'], views['View30'], views['View60'], total_score]

            dict_of_score_and_views_for_the_day[article_hash][index] = list_of_score
            dict_of_score_and_views_for_the_day[article_hash][len(organization_keys)] += dict_of_score_and_views_for_the_day[article_hash][index]


    col_names = ["is_present", "view", "view5", "view10", "view30", "view60", "score"]
    row_names = ["ledroit", "lesoleil", "lenouvelliste", "lequotidien", "latribune", "lavoixdelest", "total"]

    for article_hash in all_hashes_of_the_day:
        dict_of_score_and_views_for_the_day[article_hash] = pd.DataFrame(dict_of_score_and_views_for_the_day[article_hash], columns=col_names, index=row_names)

    return dict_of_score_and_views_for_the_day

def get_slug_from_org(p_hash):
    slug_dict = {}
    article = getArticle(p_hash)
    if article is None:
        return {}
    for org in range(len(article[0])):
        organization = article[0][org]["organizationKey"]
        slug = article[0][org]["publications"][0]["slug"]["fr"]
        slug_dict[organization] = slug.split("/")
    return slug_dict


def get_popularity_and_slug(p_dict_of_scores):
    slugs_and_score = {}
    all_hashes = p_dict_of_scores.keys()
    for hash in all_hashes:
        try:
            slugs = get_slug_from_org(hash)
            slugs_and_score[hash]["slug_word"] = slugs
            for org in slugs.keys():
                slugs_and_score[hash]["slug_word"][org] = p_dict_of_scores[hash].loc[org]
        except KeyError:
            slugs_and_score[hash] = None
        except UnicodeDecodeError:
            slugs_and_score[hash] = None
    return slugs_and_score


def slug_and_score_by_article_by_lesoleil(p_hash, p_dict_with_score_info):
    dict_score_slug = {}
    article_info = getArticle(p_hash)
    present_in_info = False
    try:
        for org in range(len(article_info[0])):
            if article_info[0][org]["organizationKey"] == "lesoleil":
                present_in_info = True
                slug = article_info[0][org]["publications"][0]["slug"]["fr"]
                slug = slug.split("/")[0]
                #slug = "_".join(slug)
                dict_score_slug["slug"] = slug
                break
    except TypeError:
        return None
    if p_dict_with_score_info[p_hash].loc["lesoleil"]["is_present"] == 0 or present_in_info != True:
        return None
    dict_score_slug["score"] =  p_dict_with_score_info[p_hash].loc["lesoleil"]["score"]
    return dict_score_slug

def extract_articles_by_day(day, train_test=True):
    articles_in_month = extract_articles_by_month(day.month, train_test)
    articles = list()

    for article in articles_in_month:
        date = article["creationDate"][:-14]
        date = datetime.strptime(date, '%Y-%m-%d')
        if date == day:
            articles.append(article)
    return articles

def extract_articles_by_month(month, train_test=True):
    if train_test:
        subPath = "/train/"
    else:
        subPath = "/test/"
    path = dataPath + subPath

    articles = list()

    for filename in os.listdir(path):
        if ("--publication-info" not in filename):
            article = getArticleByPath([path + filename])[0]
            date = article["creationDate"][:-14]
            date = datetime.strptime(date, '%Y-%m-%d')
            if date.month == month:
                articles.append(article)
    return articles


def extract_articles_range(date_start, date_end, train_test=True):
    if train_test:
        subPath = "/train"
    else:
        subPath = "/test"
    path = dataPath + subPath

    articles = list()

    for filename in os.listdir(path):
        article = getArticleByPath([filename])[0]
        date = article["creationDate"][:-14]
        date = datetime.strptime(date, '%Y-%m-%d')
        if date_start <= date < date_end:
            articles.append(article)
    return articles

def extract_articles_in_season(season, train_test=True):

    return extract_articles_range(seasons[season][0], seasons[season][1], train_test)


def cleanAuteur(auteur):
    try:
        auteur = auteur[0]['name']
        auteur = auteur.lower()
        auteur = auteur.replace('î','i')
        auteur = auteur.replace('è','e')
        auteur = auteur.replace('é','e')
        auteur = auteur.replace('ô','o')
        auteur = auteur.replace('#','')
        auteur = auteur.replace('*','')
        auteur = re.sub(r"\s*\(.*\)\s*","",auteur)
        for word in auteur.split(' '):
            if "@" in word:

                auteur = auteur.replace(word,'')

        auteur = auteur
        if ' et ' in auteur:
            auteur = auteur.split(' et ')
        elif ', ' in auteur:
            auteur = auteur.split(', ')
        elif '\r\n' in auteur:
            auteur = auteur.split('\r\n')
        else:
            auteur = [auteur]
        auteurClean = list()
        for i in auteur:
            clean = i.lstrip(' ')
            clean = clean.rstrip(' ')
            auteurClean.append(clean)
        return auteurClean
    except IndexError:
        return []


def cleanChapters(chapter):
    structure = list()
    try:
        for j in chapter:
            structure.append(j['type'])

    except TypeError:
        pass

    return structure


def cleanCreationDate(date):
    return (date.astype('datetime64[D]').view('int64') - 4) % 7

def cleanExternalId(ExternalId):
    if ExternalId == {}:
        return 0
    else:
        return 1

def cleanTitle(title):

    title = title['fr']
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

def replaceSlug(publications,equivalence):
    slug = publications[0]['slug']['fr'].split('/')[0]
    try:
        slug = equivalence[slug]
    except KeyError:
        pass
    return slug


def cleanSlug(publications, aGarder):
    if publications in aGarder:
        return publications
    return None


def mergeScore(dfInfoPublication):
    df = pd.concat([get_df_score_lesoleil(),
                    get_df_score_lavoixdelest(),
                    get_df_score_ledroit(),
                    get_df_score_latribune(),
                    get_df_score_lenouvelliste(),
                    get_df_score_lequotidien()])

    df = pd.merge(dfInfoPublication, df,how='left',left_on =['_id','organizationKey'], right_on=["hash","source"])
    return df




def getTrainingData():
    publicationsTest = get_test_publication_over_2019()
    publicationsTest = pd.DataFrame(publicationsTest)
    equivalence = {'actualite':'actualites',
                   'opinion':'opinions',
                   'essais-routiers':'auto',
                   'toit-et-moi':'maison',
                   'richardtherrien': 'arts',
                    'richard-therrien': 'arts',
                   'magazine-affaires':'affaires',
                   'claude-villeneuve':'chroniques',
                   'steve-turcotte':'sports',
                   'perspectives-economiques-2019':'affaires'}

    publicationsTest['publications'] = publicationsTest['publications'].apply(lambda x: replaceSlug(x,equivalence))

    aGarder = publicationsTest['publications'].value_counts()[publicationsTest['publications'].value_counts() > 10]

    trainArticle = get_train_article_over_2019()
    publications = get_train_publication_over_2019()
    publicationsDf = pd.DataFrame(publications)
    publicationsDf['_id'] = publicationsDf['id']
    publicationsDf = publicationsDf.drop(['id','editionId','type'],axis=1)
    publicationsDf['publications'] = publicationsDf['publications'].apply(lambda x: replaceSlug(x,equivalence))
    publicationsDf['publications'] = publicationsDf['publications'].apply(lambda x : cleanSlug(x, aGarder))
    publicationsDf = publicationsDf.dropna()
    publicationsDf = publicationsDf.drop_duplicates()


    df = pd.DataFrame(trainArticle)
    df['_id'] = df['id']
    df = df.drop(['type','templateName','canonicalUrlOverride', 'contents','visual', 'availableInPreview','lead','url','id','modificationDate'],axis=1)
    df['creationDate'] = [cleanCreationDate(i) for i in df['creationDate'].values]
    df['authors'] = df['authors'].apply(lambda x: cleanAuteur(x))
    df['chapters'] = df['chapters'].apply(lambda x: cleanChapters(x))
    df['externalIds'] = df['externalIds'].apply(lambda x: cleanExternalId(x))
    df['title'] = df['title'].apply(lambda x: cleanTitle(x))

    df = pd.merge(df, publicationsDf, how='right', on='_id')
    df = mergeScore(df)


    return df



def journalSummary(p_journal,fileQuantity = 'all'):


    journalSummary = dict()
    dateRange = pd.date_range(start = '2019-01-01', end = '2019-07-31')
    if fileQuantity =='all' :
        pass
    else:
        dateRange = dateRange[0:fileQuantity]
    for p_date in dateRange:
        p_date = (str(p_date).split(' ')[0])
        print(p_date)
        dayInfo = getDayInfo(p_date,p_journal)


        for view in dayInfo :
            try:
                journalSummary[view['hash']][view['name']] += 1
            except KeyError:
                journalSummary[view['hash']] = {'View':0,'View5':0,'View10':0,'View30':0,'View60':0}
                journalSummary[view['hash']][view['name']] += 1

    return journalSummary

if __name__ == "__main__":

    test = getArticle("ffffdb06d44290491c301af0a08a2e5b")
    test[1]["title"]["fr"]
    p_date = "2019-02-01"
    dict_info_total = get_all_scores_and_view_per_organization_for_a_day(p_date)

    testing = slug_and_score_by_article_by_lesoleil("27b803e15021b1313fb1d9750a9d65a7",
                                                    p_dict_with_score_info=dict_info_total)

    # pd.DataFrame(dict_of_score_and_views_for_the_day[article_hash], columns=col_names, index=row_names)

    hash_article_test = "d9b71cfa7d9dfcbae96e390180fd5335"
    dict_slug_score_leSoleil = {"slug": [], "score": []}
    for hash in list(dict_info_total.keys()):
        dict_value = slug_and_score_by_article_by_lesoleil(hash, p_dict_with_score_info=dict_info_total)
        if dict_value is not None:
            dict_slug_score_leSoleil["slug"].append(dict_value["slug"])
            dict_slug_score_leSoleil["score"].append(dict_value["score"])

    test_dataframe = pd.DataFrame.from_dict(dict_slug_score_leSoleil)
    test_dataframe.loc[(test_dataframe.slug == "actualite"), "slug"] = "actualites"
    # test_dataframe = test_dataframe.groupby(['slug']).reset_index()

    test_dataframe = test_dataframe.loc[
        (test_dataframe['slug'] == "actualites") or (test_dataframe['slug'] == "affaires")]

    quantile_90 = test_dataframe["score"].quantile(0.90)

    test_dataframe = test_dataframe.loc[test_dataframe['score'] < quantile_90]

    sns.boxplot(x="slug", y="score", data=test_dataframe)
    plt.show()
    # test_2 = slug_and_score_by_article_by_lesoleil(hash_article_test)

    testing_hashes = ['0deb8dafb131e6ac8775b2d973c2c306', '4abda728af6e702ffdccecc714ed2fcf',
                      '531fca26523a603539db1ebd7cb5203e', 'eb096788e215d7dd54e001d2cd4423ec',
                      '32245de2f0c9ecc1cf701f7d9ab3485c', 'ec68cd02afeb06180152a15d8cfbf9a7',
                      '465c91f8bc131f4ff55de9c9e7a5fbed', '45dbda2e61b75a1f526a968c546c8595',
                      'ae340e547032d3cefa0a4fabd23f4cbc', 'af6ba32b67c53e28988beb55cf80fab0']
    for hash in testing_hashes:
        test = getArticle(hash)
        for org in range(len(test[0])):
            print("hash: ", hash)
            print("Organisation: ", test[0][org]["organizationKey"])
            slug = test[0][org]["publications"][0]["slug"]["fr"]
            slug = slug.split("/")
            slug = slug[0]
            slug = "_".join(slug)
            print("slug: ", slug)
            print("url: ", test[1]["url"])
            print("Scores: ")
            print(dict_info_total[hash])
            print("------------------------------------")

    # score_per_months = list()
    #
    # for month in range(1, 13):
    #     articles_ion_month = extract_articles_by_month(month)
    #     for article in articles_ion_month:
    #         hash = article['hash']
    #         score_per_months.append(get_views_by_hash(hash))

    #lenouvelliste = dayPopularity(p_date, "lenouvelliste")
    #lenouvelliste["odeb8dafb131e6ac8775b2d973c2c306"]


    #test_slug_fun = get_slug_from_org("eb096788e215d7dd54e001d2cd4423ec")

    #test_slug_with_score = get_popularity_and_slug(dict_info_total)


    #test = getArticle("eb096788e215d7dd54e001d2cd4423ec")
    #dict_info_total["eb096788e215d7dd54e001d2cd4423ec"]

    #test_slug_with_score["531fca26523a603539db1ebd7cb5203e"]

    #len(test_slug_with_score)
import configparser
import json
import os
import operator
import pandas as pd
from datetime import datetime
import dateutil.parser
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)

tips = sns.load_dataset("tips")


configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

def getArtigleByPath(path):
    ArticleInfo = list()
    try:
        for file in path:
            with open(file, 'r') as article:
                ArticleInfo.append(json.load(article))

    except UnicodeDecodeError:
        return None
    except NameError:
        return None
    except FileNotFoundError:
        return None

def getArticle(p_hash):
    fileInfoPath = f'{dataPath}/train/{p_hash}'
    selectedJsonFiles = [f'{fileInfoPath}--publication-info.json',f'{fileInfoPath}.json']

    ArticleInfo = getArtigleByPath(selectedJsonFiles)

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

            total_score = views['View'] + views['View5'] + (2 * views['View10']) + (5 * views['View30']) + (10 * views['View60'])
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
    articles_in_month = extract_articles_by_month()
    articles = list()

    for article in articles_in_month:
        date = article["creationDate"][:-14]
        date = datetime.strptime(date, '%Y/%m/%d')
        if date== day:
            articles.append(article)
    return articles

def extract_articles_by_month(month, train_test=True):
    if train_test:
        subPath = "/train"
    else:
        subPath = "/test"
    path = dataPath + subPath

    articles = list()

    for filename in os.listdir(path):
        article = getArtigleByPath(filename)
        date = article["creationDate"][:-14]
        date = datetime.strptime(date, '%Y/%m/%d')
        if date.month == month:
            articles.append(article)
    return articles




if __name__ == "__main__":
    

    p_date = "2019-02-01"
    dict_info_total = get_all_scores_and_view_per_organization_for_a_day(p_date)


    testing = slug_and_score_by_article_by_lesoleil("27b803e15021b1313fb1d9750a9d65a7", p_dict_with_score_info = dict_info_total)


    #pd.DataFrame(dict_of_score_and_views_for_the_day[article_hash], columns=col_names, index=row_names)

    hash_article_test = "d9b71cfa7d9dfcbae96e390180fd5335"
    dict_slug_score_leSoleil = {"slug" : [], "score" :  [] }
    for hash in list(dict_info_total.keys())[0:1000]:
        dict_value = slug_and_score_by_article_by_lesoleil(hash, p_dict_with_score_info = dict_info_total)
        if dict_value is not None:
            dict_slug_score_leSoleil["slug"].append(dict_value["slug"])
            dict_slug_score_leSoleil["score"].append(dict_value["score"])



    test_dataframe = pd.DataFrame.from_dict(dict_slug_score_leSoleil)
    test_dataframe = test_dataframe.groupby(['slug']).sum().reset_index()

    sns.catplot(x="slug", y="score", data=test_dataframe)
    #test_2 = slug_and_score_by_article_by_lesoleil(hash_article_test)




    testing_hashes = ['0deb8dafb131e6ac8775b2d973c2c306', '4abda728af6e702ffdccecc714ed2fcf', '531fca26523a603539db1ebd7cb5203e', 'eb096788e215d7dd54e001d2cd4423ec', '32245de2f0c9ecc1cf701f7d9ab3485c', 'ec68cd02afeb06180152a15d8cfbf9a7', '465c91f8bc131f4ff55de9c9e7a5fbed', '45dbda2e61b75a1f526a968c546c8595', 'ae340e547032d3cefa0a4fabd23f4cbc', 'af6ba32b67c53e28988beb55cf80fab0']
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






    #lenouvelliste = dayPopularity(p_date, "lenouvelliste")
    #lenouvelliste["odeb8dafb131e6ac8775b2d973c2c306"]


    #test_slug_fun = get_slug_from_org("eb096788e215d7dd54e001d2cd4423ec")

    #test_slug_with_score = get_popularity_and_slug(dict_info_total)

    #test = getArticle("eb096788e215d7dd54e001d2cd4423ec")
    #dict_info_total["eb096788e215d7dd54e001d2cd4423ec"]

    #test_slug_with_score["531fca26523a603539db1ebd7cb5203e"]

    #len(test_slug_with_score)
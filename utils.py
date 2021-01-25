import json
import os
import operator
import pandas as pd


def getArticle(p_hash):
    fileInfoPath = f'data/train/{p_hash}'
    selectedJsonFiles = [f'{fileInfoPath}--publication-info.json',f'{fileInfoPath}.json']

    ArticleInfo = list()
    for file in selectedJsonFiles:
        with open(file, 'r') as article:
            ArticleInfo.append(json.load(article))
    return ArticleInfo



def getDayInfo(p_date,p_journal) :

    journalPath = f'data/analytics/{p_journal}/'

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



def dayPopularity(p_date,p_journal):

    p_dayInfoSummary = dayInfoSummary(p_date,p_journal)

    dayPopularity = dict()
    for articleHash, views in p_dayInfoSummary.items():
        dayPopularity[articleHash] = (views['View']) + (views['View5']) + (views['View10'] * 2) + (views['View30'] * 5) + (views['View60'] * 10)

    return dayPopularity



test = dayPopularity('2019-02-01','lesoleil')
test2=getArticle(max(test.items(), key=operator.itemgetter(1))[0])







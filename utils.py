import json
import os
import operator
import pandas as pd


def getArticle(p_hash):
    fileInfoPath = f'data/train/{p_hash}'
    selectedJsonFiles = [f'{fileInfoPath}--publication-info.json',f'{fileInfoPath}.json']

    ArticleInfo = list()
    for file in selectedJsonFiles:
        try:
            with open(file, 'r') as article:
                ArticleInfo.append(json.load(article))
        except FileNotFoundError:
            file = file.replace('train','test')
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




def journalSummary(p_journal):

    journalSummary = dict()
    dateRange = pd.date_range(start = '2019-01-01', end = '2019-07-31')
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




if __name__ == '__main__':
    test = dayPopularity('2019-02-01','lesoleil')
    test2=getArticle(max(test.items(), key=operator.itemgetter(1))[0])
    getArticle('d9b71cfa7d9dfcbae96e390180fd5335')[0][0]['publications'][0]['publicationDate'].find('2019') > -1


    cout = 0
    for file in [file.replace('.json','') for file in os.listdir("data/train/") if not file.endswith("-info.json")]:
        if getArticle(file)[1]['creationDate'].find('2019') > -1:
            print(getArticle(file)[1]['creationDate'])
            
            cout +=1

    len([file.replace('.json','') for file in os.listdir("data/test/") if not file.endswith("-info.json")])


    len(os.listdir("data/train/"))

    getArticle(file)[1]['creationDate']




import configparser
import json
import os
import operator
import pandas as pd
import dateutil.parser
import numpy as np

configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

def getArticle(p_hash):
    fileInfoPath = f'{dataPath}/train/{p_hash}'
    selectedJsonFiles = [f'{fileInfoPath}--publication-info.json',f'{fileInfoPath}.json']

    ArticleInfo = list()
    for file in selectedJsonFiles:
        with open(file, 'r') as article:
            ArticleInfo.append(json.load(article))
    return ArticleInfo



def getDayInfo(p_date,p_journal) :

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



if __name__ == "__main__":
    
    #creation_date_article = dateutil.parser.parse(article_info[1]["creationDate"]).strftime("%Y-%m-%d") 
    #views_and_score_by_organization_for_article = views_and_score_by_organization_for_article(creation_date_article ,hash_artile_test)
    #views_and_score_by_organization_for_article


    #organization_key = "ledroit"

    #info_for_organizationledroit = dayInfoSummary(p_date, organization_key)
    #info_for_organizationledroit = dict(list(info_for_organizationledroit.items())[0: 3]) 
    #info_for_organizationledroit

    #organization_key = "lesoleil"

    #info_for_organizationsoleil = dayInfoSummary(p_date, organization_key)
    #info_for_organizationsoleil = dict(list(info_for_organizationsoleil.items())[0: 3]) 
    #info_for_organizationsoleil
    p_date = "2019-02-01"
    dict_info_total = get_all_scores_and_view_per_organization_for_a_day(p_date)


    dict_info_total["e48d9e71949d83f3633a44e4cca242b5"]

    hash_artile_test = "d9b71cfa7d9dfcbae96e390180fd5335"
    hash_artile_test = "e48d9e71949d83f3633a44e4cca242b5"
    #article_info =getArticle(hash_artile_test)
    #dict_info_total[hash_artile_test]


    #info_for_organizationledroit["ff2d4b6b86b58f9cdc361eb5553b0480"]["View"] = 69
    #info_for_organizationledroit
    #a = {"b" : {"test" : 12}, "c" : {"test" : A"} }

    #a = {"b" : {"d" : 12}, "c" : {"d" : 15}}
    #a["c"]["d"] = "fuck off"
    #a

    #info_for_organizationledroit


    #for key, value in a.items():
    #    print(key, value)




    #numpy_array = np.zeros((8, 6))
    #numpy_array[7] = [1,2,3,4,5,6]
    #numpy_array[7] += [6,5,4,3,2,1]



    #np.random.seed(1618033)
    #
    ##Set 3 axis labels/dims
    #years = np.arange(2000,2010) #Years
    #samples = np.arange(0,20) #Samples
    #patients = np.array(["patient_%d" % i for i in range(0,3)]) #Patients
    #
    ##Create random 3D array to simulate data from dims above
    #A_3D = np.random.random((years.size, samples.size, len(patients))) #(10, 20, 3)
    #
    ## Create the MultiIndex from years, samples and patients.
    #midx = pd.MultiIndex.from_product([years, samples, patients])
    #
    ## Create sample data for each patient, and add the MultiIndex.
    #patient_data = pd.DataFrame(np.random.randn(len(midx), 3), index = midx)


    #df = pd.DataFrame(a, columns=col_names, index=row_names)
#
    #dict_score_per_article_per_organisation = {"total" : {"views" : [0] * 5, "score" : 0}}
#
    #a = pd.dataframe( ["ledroit", "lesoleil", "lenouvelliste", "lequotidien", "latribune", "lavoixdelest"])
    #a =  pd.DataFrame("organization" : ["ledroit", "lesoleil", "lenouvelliste", "lequotidien", "latribune", "lavoixdelest"])
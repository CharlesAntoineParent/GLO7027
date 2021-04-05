from pymongo import MongoClient
import configparser
import json
import os
from datetime import datetime
import dateutil.parser

configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

online_client = MongoClient("mongodb+srv://Olivier:Olivier7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority")
localhost_client = MongoClient(port=27017)


db_online = online_client.get_database("GLO-7027-13")
db_local =localhost_client.get_database("GLO-7027-13")




#score_pipeline = [
    {
        '$project': {
            'hash': 1, 
            'name': 1
        }
    }, {
        '$group': {
            '_id': '$hash', 
            'count': {
                '$sum': 1
            }, 
            'view': {
                '$sum': {
                    '$cond': [
                        {
                            '$eq': [
                                '$name', 'View'
                            ]
                        }, 1, 0
                    ]
                }
            }, 
            'view5': {
                '$sum': {
                    '$cond': [
                        {
                            '$eq': [
                                '$name', 'View5'
                            ]
                        }, 1, 0
                    ]
                }
            }, 
            'view10': {
                '$sum': {
                    '$cond': [
                        {
                            '$eq': [
                                '$name', 'View10'
                            ]
                        }, 1, 0
                    ]
                }
            }, 
            'view30': {
                '$sum': {
                    '$cond': [
                        {
                            '$eq': [
                                '$name', 'View30'
                            ]
                        }, 1, 0
                    ]
                }
            }, 
            'view60': {
                '$sum': {
                    '$cond': [
                        {
                            '$eq': [
                                '$name', 'View60'
                            ]
                        }, 1, 0
                    ]
                }
            }, 
            'score': {
                '$sum': {
                    '$switch': {
                        'branches': [
                            {
                                'case': {
                                    '$eq': [
                                        '$name', 'View'
                                    ]
                                }, 
                                'then': 1
                            }, {
                                'case': {
                                    '$eq': [
                                        '$name', 'View5'
                                    ]
                                }, 
                                'then': 1
                            }, {
                                'case': {
                                    '$eq': [
                                        '$name', 'View10'
                                    ]
                                }, 
                                'then': 2
                            }, {
                                'case': {
                                    '$eq': [
                                        '$name', 'View30'
                                    ]
                                }, 
                                'then': 5
                            }, {
                                'case': {
                                    '$eq': [
                                        '$name', 'View60'
                                    ]
                                }, 
                                'then': 10
                            }
                        ], 
                        'default': 0
                    }
                }
            }
        }
    }
#]
#
#analytics_lesoleil_local = db_local.get_collection("analytics_lesoleil")
#analytics_ledroit_local = db_local.get_collection("analytics_ledroit")
#analytics_lenouvelliste_local = db_local.get_collection("analytics_lenouvelliste")
#analytics_lequotidien_local = db_local.get_collection("analytics_lequotidien")
#analytics_latribune_local = db_local.get_collection("analytics_latribune")
#analytics_lavoixdelest_local = db_local.get_collection("analytics_lavoixdelest")

#db_online.get_collection("score_lesoleil").insert_many(analytics_lesoleil_local.aggregate(score_pipeline, allowDiskUse =True))
#db_online.get_collection("score_ledroit").insert_many(analytics_ledroit_local.aggregate(score_pipeline, allowDiskUse =True))
#db_online.get_collection("score_lenouvelliste").insert_many(analytics_lenouvelliste_local.aggregate(score_pipeline, allowDiskUse =True))
#db_online.get_collection("score_lequotidien").insert_many(analytics_lequotidien_local.aggregate(score_pipeline, allowDiskUse =True))
#db_online.get_collection("score_latribune").insert_many(analytics_latribune_local.aggregate(score_pipeline, allowDiskUse =True))
#db_online.get_collection("score_lavoixdelest").insert_many(analytics_lavoixdelest_local.aggregate(score_pipeline, allowDiskUse =True))




def get_article_by_path(path):
            with open(path, encoding='utf-8') as article:
                return json.load(article)

def add_hashid_to_publication_info(p_json_file, p_hash):
    for index in range(len(p_json_file)):
        p_json_file[index]["id"] = p_hash
    return p_json_file

def format_train_publication_info(filename):
    train_path = f'{dataPath}/train/'
    file_path = train_path + filename
    publication_info = get_article_by_path(file_path)
    hash_article = filename.split('-')[0]
    publication_info_with_id = add_hashid_to_publication_info(publication_info, hash_article)
    for i in range(len(publication_info_with_id)):
        for j in range(len(publication_info_with_id[i]["publications"])):
            publication_info_with_id[i]["publications"][j]["publicationDate"] = dateutil.parser.parse(publication_info_with_id[i]["publications"][j]["publicationDate"], ignoretz=True)

    return publication_info_with_id

def format_train_article_info(filename):
    train_path = f'{dataPath}/train/'
    file_path = train_path + filename
    article_info = get_article_by_path(file_path)
    article_info["creationDate"] = dateutil.parser.parse(article_info["creationDate"], ignoretz=True)
    article_info["modificationDate"] = dateutil.parser.parse(article_info["modificationDate"], ignoretz=True)
    return [article_info]

def format_test_publication_info(filename):
    test_path = f'{dataPath}/test/'
    file_path = test_path + filename
    publication_info = get_article_by_path(file_path)
    hash_article = filename.split('-')[0]
    publication_info_with_id = add_hashid_to_publication_info(publication_info, hash_article)
    for i in range(len(publication_info_with_id)):
        for j in range(len(publication_info_with_id[i]["publications"])):
            publication_info_with_id[i]["publications"][j]["publicationDate"] = dateutil.parser.parse(publication_info_with_id[i]["publications"][j]["publicationDate"], ignoretz=True)

    return publication_info_with_id

def format_test_article_info(filename):
    test_path = f'{dataPath}/test/'
    file_path = test_path + filename
    article_info = get_article_by_path(file_path)
    article_info["creationDate"] = dateutil.parser.parse(article_info["creationDate"], ignoretz=True)
    article_info["modificationDate"] = dateutil.parser.parse(article_info["modificationDate"], ignoretz=True)
    return [article_info]


train_path = f'{dataPath}/train/'
for index, filename in enumerate(os.listdir(train_path)):
    if "--publication-info.json" in filename:
        db_online.get_collection("train_publication_info").insert_many(format_train_publication_info(filename))
    else:
        db_online.get_collection("train_article_info").insert_many(format_train_article_info(filename))






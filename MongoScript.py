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




if __name__ == '__main__':



    #client = MongoClient("mongodb+srv://Charles:Charles7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") #Charles
    #client = MongoClient("mongodb+srv://Vincent :Vincent7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") Vincent
    #client = MongoClient("mongodb+srv://Olivier:Olivier7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") # Olivier

    #analytics = db.get_collection("analytics")

    #organisations = ["ledroit","lesoleil", "lenouvelliste", "lequotidien", "latribune", "lavoixdelest"]
    #for organisation in organisations:
    #    print(organisation)
    #    analyticPath = f'{dataPath}/analytics/{organisation}'
    #    for index, filename in enumerate(os.listdir(analyticPath)):
    #        now = datetime.now()
    #        current_time = now.strftime("%H:%M:%S")
    #        print("Article =", index, "Current Time =", current_time)
    #        path_to_file = analyticPath + "/" + filename
    #        with open(path_to_file, encoding='utf-8') as file:
    #            json_file = json.load(file)
#
    #        analytics.insert_many(json_file)
#
    

    



    




    

    

    #publication_info = get_publication_info("0000514628aeedb83e30f922ea4bd552")
  

    #train_article_publication_info = format_train_article_info("ffffdb06d44290491c301af0a08a2e5b--publication-info.json")
    #test = get_publication_info("000a1ec0602dcbaf97bed2755cbc19c4--publication-info.json")
    #train_article_info = db.get_collection("train_article_info")

    
    


    client = MongoClient(port=27017)
    db=client.get_database("GLO-7027-13")
    train_publication_info = db.get_collection("train_publication_info")
    train_article_info = db.get_collection("train_article_info")

    test_publication_info = db.get_collection("test_publication_info")
    test_article_info = db.get_collection("test_article_info")


    train_path = f'{dataPath}/train/'
    for index, filename in enumerate(os.listdir(train_path)):
        if "--publication-info.json" in filename:
            train_publication_info.insert_many(format_train_publication_info(filename))
        else:
            train_article_info.insert_many(format_train_article_info(filename))

    test_path = f'{dataPath}/test/'
    for index, filename in enumerate(os.listdir(test_path)):
        if "--publication-info.json" in filename:
            test_publication_info.insert_many(format_test_publication_info(filename))
        else:
            test_article_info.insert_many(format_test_article_info(filename))





    #train_article_publication_info = format_train_publication_info("ffffdb06d44290491c301af0a08a2e5b--publication-info.json")
    #train_article_publication_info[0]["publications"][0]["publicationDate"] = dateutil.parser.parse(train_article_publication_info[0]["publications"][0]["publicationDate"], ignoretz=True)


    #test  =dateutil.parser.parse(train_article_publication_info[0]["publications"][0]["publicationDate"], ignoretz=True)




    #train_publication_info.insert_many(train_article_publication_info)

    #train_article_info_toInsert = format_train_article_info("ffffdb06d44290491c301af0a08a2e5b.json")
    #train_article_info_toInsert[0]["creationDate"] = dateutil.parser.parse(train_article_publication_info[0]["creationDate"], ignoretz=True)
    #train_article_info_toInsert[0]["modificationDate"] = dateutil.parser.parse(train_article_publication_info[0]["modificationDate"], ignoretz=True)

    #train_article_info.insert_many([train_article_info_toInsert])
    #train_publication_info.insert_many(publication_info_with_id)


    

    analytics_ledroit = db.get_collection("analytics_ledroit")


    analyticPath = f'{dataPath}/analytics/ledroit'
    for index, filename in enumerate(os.listdir(analyticPath)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Article =", index, "Current Time =", current_time)
        path_to_file = analyticPath + "/" + filename
        with open(path_to_file, encoding='utf-8') as file:
            json_file = json.load(file)
        
        for i in range(len(json_file)):
            json_file[i]["createdAt"] = dateutil.parser.parse(json_file[i]["createdAt"] , ignoretz=True)

        analytics_ledroit.insert_many(json_file)


    analytics_lesoleil = db.get_collection("analytics_lesoleil")

    analyticPath = f'{dataPath}/analytics/lesoleil'
    for index, filename in enumerate(os.listdir(analyticPath)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Article =", index, "Current Time =", current_time)
        path_to_file = analyticPath + "/" + filename
        with open(path_to_file, encoding='utf-8') as file:
            json_file = json.load(file)

        for i in range(len(json_file)):
            json_file[i]["createdAt"] = dateutil.parser.parse(json_file[i]["createdAt"] , ignoretz=True)

        analytics_lesoleil.insert_many(json_file)


    analytics_lenouvelliste = db.get_collection("analytics_lenouvelliste")

    analyticPath = f'{dataPath}/analytics/lenouvelliste'
    for index, filename in enumerate(os.listdir(analyticPath)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Article =", index, "Current Time =", current_time)
        path_to_file = analyticPath + "/" + filename
        with open(path_to_file, encoding='utf-8') as file:
            json_file = json.load(file)
        
        for i in range(len(json_file)):
            json_file[i]["createdAt"] = dateutil.parser.parse(json_file[i]["createdAt"] , ignoretz=True)
        
        analytics_lenouvelliste.insert_many(json_file)



    analytics_lequotidien = db.get_collection("analytics_lequotidien")

    analyticPath = f'{dataPath}/analytics/lequotidien'
    for index, filename in enumerate(os.listdir(analyticPath)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Article =", index, "Current Time =", current_time)
        path_to_file = analyticPath + "/" + filename
        with open(path_to_file, encoding='utf-8') as file:
            json_file = json.load(file)
    
        for i in range(len(json_file)):
            json_file[i]["createdAt"] = dateutil.parser.parse(json_file[i]["createdAt"] , ignoretz=True)
        
        analytics_lequotidien.insert_many(json_file)



    analytics_latribune = db.get_collection("analytics_latribune")

    analyticPath = f'{dataPath}/analytics/latribune'
    for index, filename in enumerate(os.listdir(analyticPath)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Article =", index, "Current Time =", current_time)
        path_to_file = analyticPath + "/" + filename
        with open(path_to_file, encoding='utf-8') as file:
            json_file = json.load(file)

        for i in range(len(json_file)):
            json_file[i]["createdAt"] = dateutil.parser.parse(json_file[i]["createdAt"] , ignoretz=True)
        
        analytics_latribune.insert_many(json_file)


    analytics_lavoixdelest = db.get_collection("analytics_lavoixdelest")

    analyticPath = f'{dataPath}/analytics/lavoixdelest'
    for index, filename in enumerate(os.listdir(analyticPath)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Article =", index, "Current Time =", current_time)
        path_to_file = analyticPath + "/" + filename
        with open(path_to_file, encoding='utf-8') as file:
            json_file = json.load(file)
        
        for i in range(len(json_file)):
            json_file[i]["createdAt"] = dateutil.parser.parse(json_file[i]["createdAt"] , ignoretz=True)

        analytics_lavoixdelest.insert_many(json_file)





score_pipeline = [
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
]





analytics_ledroit = db.get_collection("analytics_ledroit")


    analyticPath = f'{dataPath}/analytics/ledroit'
    for index, filename in enumerate(os.listdir(analyticPath)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Article =", index, "Current Time =", current_time)
        path_to_file = analyticPath + "/" + filename
        with open(path_to_file, encoding='utf-8') as file:
            json_file = json.load(file)
        
        for i in range(len(json_file)):
            json_file[i]["createdAt"] = dateutil.parser.parse(json_file[i]["createdAt"] , ignoretz=True)

        analytics_ledroit.insert_many(json_file)

db.get_collection("score_ledroit").insert_many(analytics_ledroit.aggregate(score_pipeline))









db.get_collection("score_lesoleil").insert_many(analytics_lesoleil.aggregate(score_pipeline))
db.get_collection("score_lenouvelliste").insert_many(analytics_lenouvelliste.aggregate(score_pipeline))
db.get_collection("score_lequotidien").insert_many(analytics_lequotidien.aggregate(score_pipeline))
db.get_collection("score_latribune").insert_many(analytics_latribune.aggregate(score_pipeline))
db.get_collection("score_lavoixdelest").insert_many(analytics_lavoixdelest.aggregate(score_pipeline))




















    client.close()
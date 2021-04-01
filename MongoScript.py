from pymongo import MongoClient
import configparser
import json
import os
from datetime import datetime

if __name__ == '__main__':

    

    configParser = configparser.RawConfigParser()
    configFilePath = r'config.txt'
    configParser.read(configFilePath)
    dataPath = configParser.get('config', 'dataPath')


    #client = MongoClient("mongodb+srv://Charles:Charles7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") #Charles
    #client = MongoClient("mongodb+srv://Vincent :Vincent7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") Vincent
    client = MongoClient("mongodb+srv://Olivier:Olivier7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") # Olivier
    
    client = MongoClient(port=27017)
    db=client.get_database("GLO-7027-13")
    analytics = db.get_collection("analytics")

    organisations = ["ledroit","lesoleil", "lenouvelliste", "lequotidien", "latribune", "lavoixdelest"]
    for organisation in organisations:
        print(organisation)
        analyticPath = f'{dataPath}/analytics/{organisation}'
        for index, filename in enumerate(os.listdir(analyticPath)):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Article =", index, "Current Time =", current_time)
            path_to_file = analyticPath + "/" + filename
            with open(path_to_file, encoding='utf-8') as file:
                json_file = json.load(file)

            analytics.insert_many(json_file)

    client.close()

import configparser
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from collections_utilis import connect_GLO7027

configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')


def df_external_id_with_hash():
    db_online = connect_GLO7027()
    train_article_collection = db_online.get_collection("train_article_info")
    train_article_external_id = train_article_collection.find({"creationDate" : {"$not" :{"$lt": datetime(2019, 1, 1, 0, 0, 0)}}}, {"id":1, "externalIds":1} )
    df_externalid = pd.json_normalize(list(train_article_external_id), meta = ["id", "externalIds"]).rename(columns = {"id": "hash"}).drop("_id", axis =1)
    df_externalid[["externalIds.newscycle"]] = df_externalid[["externalIds.newscycle"]].where(df_externalid[ ["externalIds.newscycle"]].isnull(), 1).fillna(0).astype(int)
    return df_externalid

if __name__ == '__main__':
    df_external_id = df_external_id_with_hash()
    df_external_id["externalIds.newscycle"].value_counts(normalize=True).plot(kind = "barh")
    plt.title("Distribution externalIds.newscycle", fontsize=20)
    plt.xlabel("Fréquence", fontsize=20)
    plt.ylabel("booléen externalIds.newscycle" , fontsize=20)
    plt.show()
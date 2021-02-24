import configparser
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from collections_utilis import connect_GLO7027

configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

def get_channel_with_hash():
    db_online = connect_GLO7027()
    train_article_collection = db_online.get_collection("train_article_info")
    train_article = train_article_collection.find({"creationDate" : {"$not" :{"$lt": datetime(2019, 1, 1, 0, 0, 0)}}}, {"id":1, "channel":1} )
    df_channel = pd.json_normalize(list(train_article), meta = ["id", "channel"])
    df_channel = df_channel.rename(columns = {"id": "hash", "channel.fr": "channel"})[['hash','channel']]
    df_channel["channel"] = df_channel["channel"].apply(lambda x: "_".join(str(x).split(" ")))
    return df_channel

if __name__ == '__main__':

    df_channel = get_channel_with_hash()

    f, ax = plt.subplots(figsize=(7, 6))
    ax = df_channel["channel"].value_counts(normalize=True).head(6).plot(kind = "barh")
    plt.title("Fréquences des valeurs de channel", fontsize=20)
    plt.xlabel("Fréquence", fontsize=18)
    plt.ylabel("Channel", fontsize=18)
    ax.xaxis.grid(True)
    ax.set_yticklabels(list(df_channel["channel"].value_counts(normalize=True).head(6).keys()), rotation=0, fontsize=22)
    plt.show()



        f, ax = plt.subplots(figsize=(7, 6))
    ax = df_slug["slug_1"].value_counts(normalize=True).head(6).plot(kind = "barh")
    plt.title("Fréquence des slugs des différentes sources", fontsize=20)
    plt.ylabel("Slug", fontsize=18)
    plt.xlabel("Fréquence", fontsize=18)
    ax.xaxis.grid(True)
    ax.set_yticklabels(list(df_slug["slug_1"].value_counts(normalize=True).head(6).keys()), rotation=0, fontsize=16)
    plt.show()



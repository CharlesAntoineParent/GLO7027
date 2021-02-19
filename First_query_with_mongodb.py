from pymongo import MongoClient
import configparser
import json
import os
from datetime import datetime, date
import dateutil.parser
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
import numpy as np
import math




configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

#online_client = MongoClient("mongodb+srv://Charles:Charles7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") #Charles
#online_client = MongoClient("mongodb+srv://Vincent :Vincent7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") #Vincent
online_client = MongoClient("mongodb+srv://Olivier:Olivier7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") # Olivier

# Connexion DB
db_online =online_client.get_database("GLO-7027-13")

# Connexion Collections globales
train_article_collection = db_online.get_collection("train_article_info")
train_publication_collection = db_online.get_collection("train_publication_info")

train_article = train_article_collection.find({"creationDate": { "$gte": datetime('2019-01-01  ') } }, {"type" : 0, "templateName" : 0, "visual" : 0, "availableInPreview" : 0, "url" : 0, })
train_publication = train_publication_collection.find({"creationDate": { "$gte": date.fromisoformat('2019-01-01') } }, {"id" : 1, "publications":1})

df_channel = pd.json_normalize(list(train_article), meta = ["id", "channel"]).rename(columns = {"id": "hash"}).drop("_id", axis =1)
# Connexion ledroit
score_ledroit_collection = db_online.get_collection("score_ledroit")
score_ledroit = score_ledroit_collection.find({}, {"_id" : 1, "score":1})

# Connexion lesoleil
score_lesoleil_collection = db_online.get_collection("score_lesoleil")
score_lesoleil = score_lesoleil_collection.find({}, {"_id" : 1, "score":1})

# Connexion lenouvelliste
score_lenouvelliste_collection = db_online.get_collection("score_lenouvelliste")
score_lenouvelliste = score_lenouvelliste_collection.find({}, {"_id" : 1, "score":1})

# Connexion lequotidien
score_lequotidien_collection = db_online.get_collection("score_lequotidien")
score_lequotidien = score_lequotidien_collection.find({}, {"_id" : 1, "score":1})

# Connexion latribune
score_latribune_collection = db_online.get_collection("score_latribune")
score_latribune = score_latribune_collection.find({}, {"_id" : 1, "score":1})

# Connexion lavoixdelest
score_lavoixdelest_collection = db_online.get_collection("score_lavoixdelest")
score_lavoixdelest = score_lavoixdelest_collection.find({}, {"_id" : 1, "score":1})


df_score_ledroit = pd.DataFrame(list(score_ledroit)).rename(columns = {"_id": "hash"})
df_score_lesoleil = pd.DataFrame(list(score_lesoleil)).rename(columns = {"_id": "hash"})
df_score_lenouvelliste = pd.DataFrame(list(score_lenouvelliste)).rename(columns = {"_id": "hash"})
df_score_lequotidien = pd.DataFrame(list(score_lequotidien)).rename(columns = {"_id": "hash"})
df_score_latribune = pd.DataFrame(list(score_latribune)).rename(columns = {"_id": "hash"})
df_score_lavoixdelest = pd.DataFrame(list(score_lavoixdelest)).rename(columns = {"_id": "hash"})

def external_id_creator(train_article_cursor):
    df_externalid = pd.json_normalize(list(train_article_cursor), meta = ["id", "externalIds", "channel"]).rename(columns = {"id": "hash"}).drop("_id", axis =1)
    df_externalid[["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]] = df_externalid[["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]].where(df_externalid[ ["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]].isnull(), 1).fillna(0).astype(int)
    return df_externalid

def merge_df_with_score_on_hash_key(df_with_info, df_score):
    return pd.merge(df_with_info, df_score, on="hash")

df_externalid = external_id_creator(train_article)


df_externalid_score_ledroit = merge_df_with_score_on_hash_key(df_externalid, df_score_ledroit)
df_externalid_score_lesoleil = merge_df_with_score_on_hash_key(df_externalid, df_score_lesoleil)
df_externalid_score_lenouvelliste = merge_df_with_score_on_hash_key(df_externalid, df_score_lenouvelliste)
df_externalid_score_lequotidien = merge_df_with_score_on_hash_key(df_externalid, df_score_lequotidien)
df_externalid_score_latribune = merge_df_with_score_on_hash_key(df_externalid, df_score_latribune)
df_externalid_score_lavoixdelest = merge_df_with_score_on_hash_key(df_externalid, df_score_lavoixdelest)


sns.boxplot(x="externalIds.newscycle", y="score", hue="externalIds.WCM_ARTICLE_ID" , data=df_externalid_score_ledroit, showfliers=False)
plt.show()


df_externalid_score_ledroit_describe = df_externalid_score_ledroit.groupby(["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]).describe()
df_externalid_score_lesoleil_describe = df_externalid_score_lesoleil.groupby(["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]).describe()
df_externalid_score_lenouvelliste_describe = df_externalid_score_lenouvelliste.groupby(["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]).describe()
df_externalid_score_lequotidien_describe = df_externalid_score_lequotidien.groupby(["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]).describe()
df_externalid_score_latribune_describe = df_externalid_score_latribune.groupby(["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]).describe()
df_externalid_score_lavoixdelest_describe = df_externalid_score_lavoixdelest.groupby(["externalIds.WCM_ARTICLE_ID", "externalIds.newscycle"]).describe()
    


df_slug = pd.json_normalize(list(train_publication), meta = ["id"], record_path = ["publications"]).drop(["slug.en", "publicationDate","sectionId", "type", "canonical"], axis=1).rename(columns = {"id": "hash"})
df_slug["slug.fr_2"] = df_slug["slug.fr"].apply(lambda x: "/".join(x.split("/")[:max(1, len(x.split("/"))-1)] ))
df_slug["slug.fr_1"] = df_slug["slug.fr"].apply(lambda x: x.split("/")[0] )

df_slug_score_ledroit = merge_df_with_score_on_hash_key(df_slug, df_score_ledroit)
df_slug_score_lesoleil = merge_df_with_score_on_hash_key(df_slug, df_score_lesoleil)
df_slug_score_lenouvelliste = merge_df_with_score_on_hash_key(df_slug, df_score_lenouvelliste)
df_slug_score_lequotidien = merge_df_with_score_on_hash_key(df_slug, df_score_lequotidien)
df_slug_score_latribune = merge_df_with_score_on_hash_key(df_slug, df_score_latribune)
df_slug_score_lavoixdelest = merge_df_with_score_on_hash_key(df_slug, df_score_lavoixdelest)


#df_slug_score.groupby(["slug.fr_2"]).describe().reset_index()
df_slug1_score_describe_ledroit = df_slug_score_ledroit.groupby(["slug.fr_1"])["score"].describe().reset_index()
df_slug1_score_describe_ledroit = df_slug1_score_describe_ledroit.sort_values('count', ascending=False)

list_of_top_slugs = list(df_slug1_score_describe_ledroit["slug.fr_1"].head(7))

df_slug_score_ledroit_keep_head_slug = df_slug_score_ledroit[df_slug_score_ledroit["slug.fr_1"].isin(list_of_top_slugs)]
df_slug_score_lesoleil_keep_head_slug = df_slug_score_lesoleil[df_slug_score_lesoleil["slug.fr_1"].isin(list_of_top_slugs)]
df_slug_score_lenouvelliste_keep_head_slug = df_slug_score_lenouvelliste[df_slug_score_lenouvelliste["slug.fr_1"].isin(list_of_top_slugs)]
df_slug_score_lequotidien_keep_head_slug = df_slug_score_lequotidien[df_slug_score_lequotidien["slug.fr_1"].isin(list_of_top_slugs)]
df_slug_score_latribune_keep_head_slug = df_slug_score_latribune[df_slug_score_latribune["slug.fr_1"].isin(list_of_top_slugs)]
df_slug_score_lavoixdelest_keep_head_slug = df_slug_score_lavoixdelest[df_slug_score_lavoixdelest["slug.fr_1"].isin(list_of_top_slugs)]

df_slug_score_ledroit_keep_head_slug["source"] = "ledroit"
df_slug_score_lesoleil_keep_head_slug["source"] = "lesoleil" 
df_slug_score_lenouvelliste_keep_head_slug["source"] = "lenouvelliste" 
df_slug_score_lequotidien_keep_head_slug["source"] = "lequotidien" 
df_slug_score_latribune_keep_head_slug["source"] = "latribune" 
df_slug_score_lavoixdelest_keep_head_slug["source"] = "lavoixdelest" 

total = [df_slug_score_ledroit_keep_head_slug, df_slug_score_lesoleil_keep_head_slug, df_slug_score_lenouvelliste_keep_head_slug, df_slug_score_lequotidien_keep_head_slug, df_slug_score_latribune_keep_head_slug, df_slug_score_lavoixdelest_keep_head_slug]

df_slug_score_total_keep_head_slug = pd.concat(total)


df_slug1_score_describe_lesoleil = df_slug_score_lesoleil.groupby(["slug.fr_1"]).describe()
df_slug1_score_describe_lenouvelliste = df_slug_score_lenouvelliste.groupby(["slug.fr_1"]).describe()
df_slug1_score_describe_lequotidien = df_slug_score_lequotidien.groupby(["slug.fr_1"]).describe()
df_slug1_score_describe_latribune = df_slug_score_latribune.groupby(["slug.fr_1"]).describe()
df_slug1_score_describe_lavoixdelest = df_slug_score_lavoixdelest.groupby(["slug.fr_1"]).describe()




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
from scipy.stats import ks_2samp




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

train_article = train_article_collection.find({"creationDate" : {"$not" :{"$lt": datetime(2019, 1, 1, 0, 0, 0)}}}, {"id" : 1, "channel" :1}     )
train_publication = train_publication_collection.find({"publications.publicationDate" :{"$not": {"$lt": datetime(2019, 1, 1, 0, 0, 0)}}}    )

#df_channel = pd.json_normalize(list(train_article), meta = ["id", "channel"]).rename(columns = {"id": "hash"}).drop("_id", axis =1)
# Connexion ledroit
score_ledroit_collection = db_online.get_collection("score_ledroit")
score_ledroit = score_ledroit_collection.find({})
#score_ledroit = score_ledroit_collection.find({}, {"_id" : 1, "score":1})

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
    


df_channel = pd.json_normalize(list(train_article), meta = ["id", "channel"]).rename(columns = {"id": "hash"}).drop("_id", axis =1).rename(columns = {"channel.fr": "channel"})
df_channel["channel"] = df_channel["channel"].apply(lambda x: "_".join(str(x).split(" ")))


df_slug_from_mongo = pd.json_normalize(list(train_publication), meta = ["id", "organizationKey"], record_path = ["publications"]).drop(["slug.en","sectionId", "type", "canonical"], axis=1).rename(columns = {"id": "hash", "slug.fr":"slug"})
#df_slug["slug.fr_2"] = df_slug["slug.fr"].apply(lambda x: "/".join(x.split("/")[:max(1, len(x.split("/"))-1)] ))
df_slug = df_slug_from_mongo
df_slug["slug_1"] = df_slug["slug"].apply(lambda x: x.split("/")[0] )

df_slug.loc[df_slug['slug_1'] == "actualite", 'slug_1'] = "actualites"




df_slug_ledroit = df_slug[df_slug["organizationKey"] == "ledroit"]
df_slug_lesoleil = df_slug[df_slug["organizationKey"] == "lesoleil"] 
df_slug_lenouvelliste = df_slug[df_slug["organizationKey"] == "lenouvelliste"] 
df_slug_lequotidien = df_slug[df_slug["organizationKey"] == "lequotidien"] 
df_slug_latribune = df_slug[df_slug["organizationKey"] == "latribune"] 
df_slug_lavoixdelest = df_slug[df_slug["organizationKey"] == "lavoixdelest"] 

#df_slug_ledroit_count_slug = df_slug_ledroit.groupby("slug_1").count("slug")
df_slug_ledroit_count_slug = df_slug_ledroit.groupby('slug_1')["slug"].count().reset_index()

df_slug_ledroit_count_slug.hist()

fig, axes = plt.subplots(2, 3)
plt.subplots_adjust(wspace=0.5)
fig.suptitle("Fréquence du premier terme des slugs selon les sources", fontsize=16)

df_slug_ledroit["slug_1"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[0,0])
axes[0,0].set_title("Le Droit")
axes[0,0].set_xlabel("Fréquence")
axes[0,0].set_ylabel("Slug")
df_slug_lesoleil["slug_1"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[0,1])
axes[0,1].set_title("Le Soleil")
axes[0,1].set_xlabel("Fréquence")
axes[0,1].set_ylabel("Slug")
df_slug_lenouvelliste["slug_1"].value_counts(normalize=True).head(10).plot(kind = "barh",ax = axes[0,2])
axes[0,2].set_title("Le Nouvelliste")
axes[0,2].set_xlabel("Fréquence")
axes[0,2].set_ylabel("Slug")
df_slug_lequotidien["slug_1"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[1,0])
axes[1,0].set_title("Le Quotidien")
axes[1,0].set_xlabel("Fréquence")
axes[1,0].set_ylabel("Slug")
df_slug_latribune["slug_1"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[1,1])
axes[1,1].set_title("La Tribune")
axes[1,1].set_xlabel("Fréquence")
axes[1,1].set_ylabel("Slug")
df_slug_lavoixdelest["slug_1"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[1,2])
axes[1,2].set_title("La Voix de l'Est")
axes[1,2].set_xlabel("Fréquence")
axes[1,2].set_ylabel("Slug")

plt.show()

#df_actualite = df_slug[df_slug["slug.fr_1"] == "actualite"]
#df_actualites = df_slug[df_slug["slug.fr_1"] == "actualites"]
#df_actualite= df_actualite.groupby([pd.Grouper(key='publicationDate', freq='W'), "organizationKey"]).agg(lambda x: len(list(x))).reset_index().sort_values(by='publicationDate')

df_slug_grouped_by_slug= df_slug.groupby([pd.Grouper(key='publicationDate', freq='M'), "slug_1"]).count().reset_index().sort_values(by='publicationDate')
df_slug_grouped_by_slug.plot(style='.-', title = "slug")
plt.show()

df_slug_grouped_by_source= df_slug.groupby([pd.Grouper(key='publicationDate', freq='W'), "organizationKey"]).count().reset_index().sort_values(by='publicationDate')

#dates_actualite = df_actualite.publicationDate
#values_actualite = df_actualite["slug.fr"]
#source_actualite = df_actualite.organizationKey

dates_slug = df_slug_grouped_by_source.publicationDate
values_slug= df_slug_grouped_by_source["slug"]
source_slug = df_slug_grouped_by_source.organizationKey



#df_actualite = pd.DataFrame(dict(dates_actualite=dates_actualite, values_actualite=values_actualite, source_actualite=source_actualite))
df_slug_grouped_by_source_to_plot = pd.DataFrame(dict(dates_slug=dates_slug, values_slug=values_slug, source_slug=source_slug))



fig, ax = plt.subplots()
source = {"ledroit":"ledroit",
"lesoleil" : "lesoleil",
"lenouvelliste" : "lenouvelliste",
"lequotidien" : "lequotidien",
"latribune" : "latribune",
"lavoixdelest" : "lavoixdelest"}

#df_slug_to_plot.set_index(['source_actualites', 'dates_actualites']).unstack('source_actualites')['values_actualites'].plot(style='o')
#plt.ylabel('values_slug')
#plt.show()

#plt.show()

#fig, axes = plt.subplots(1, 1)
df_slug_grouped_by_source_to_plot.set_index(['source_slug', 'dates_slug']).unstack('source_slug')['values_slug'].plot(style='.-', title = "slug")
#ax[0].set_title("actualite")
#df_slug_to_plot.set_index(['source_actualite', 'dates_actualite']).unstack('source_actualite')['values_actualite'].plot(style='o', ax=axes[1], title = "actualites")
#ax[1].set_title("actualites")
plt.show()

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


sns.boxplot(x="slug.fr_1", y="score", hue="source" , data=df_slug_score_total_keep_head_slug, showfliers=False)
plt.show()

sns.boxplot(x="source", y="score", hue="slug.fr_1"  , data=df_slug_score_total_keep_head_slug, showfliers=False)
plt.show()




test = df_slug_score_lesoleil.groupby(["slug.fr_1"]).agg(lambda x: list(x)).reset_index()


actualite_soleil = test["score"][0]
actualites_soleil = test["score"][1]
test = ks_2samp(actualite_soleil, actualites_soleil)
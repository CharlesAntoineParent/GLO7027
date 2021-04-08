from pymongo import MongoClient
from datetime import datetime
import pandas as pd

def connect_client():
#Connexion Serveur Client
    return MongoClient("mongodb+srv://Olivier:Olivier7027@cluster0.y7acq.mongodb.net/GLO-7027-13?retryWrites=true&w=majority") # Olivier

# Connexion DB
def connect_GLO7027():
    online_client = connect_client()
    return online_client.get_database("GLO-7027-13")


def get_train_article_over_2019():
    db_online = connect_GLO7027()
    train_article_collection = db_online.get_collection("train_article_info")
    train_article = train_article_collection.find({"creationDate" : {"$not" :{"$lt": datetime(2019, 1, 1, 0, 0, 0)}}}, )
    return train_article


def get_train_publication_over_2019():
    db_online = connect_GLO7027()
    train_publication_collection = db_online.get_collection("train_publication_info")
    train_publication = train_publication_collection.find({"publications.publicationDate" :{"$not": {"$lt": datetime(2019, 1, 1, 0, 0, 0)}}}    )
    return train_publication

def get_test_publication_over_2019():
    db_online = connect_GLO7027()
    train_publication_collection = db_online.get_collection("test_publication_info")
    train_publication = train_publication_collection.find({"publications.publicationDate" :{"$not": {"$lt": datetime(2019, 1, 1, 0, 0, 0)}}}    )
    return train_publication


def get_df_score_ledroit():
    db_online = connect_GLO7027()
    score_ledroit_collection = db_online.get_collection("score_ledroit")
    score_ledroit = score_ledroit_collection.find({})
    df_score_ledroit = pd.DataFrame(list(score_ledroit)).rename(columns = {"_id": "hash"})
    df_score_ledroit["source"] = "ledroit"
    df_score_ledroit["point_view"] = df_score_ledroit["view"]
    df_score_ledroit["point_view5"] = df_score_ledroit["view5"]
    df_score_ledroit["point_view10"] = df_score_ledroit["view10"] * 2
    df_score_ledroit["point_view30"] = df_score_ledroit["view30"] * 5
    df_score_ledroit["point_view60"] = df_score_ledroit["view60"] * 10
    
    return df_score_ledroit

def get_df_score_lesoleil():
    db_online = connect_GLO7027()
    score_lesoleil_collection = db_online.get_collection("score_lesoleil")
    score_lesoleil = score_lesoleil_collection.find({})
    df_score_lesoleil = pd.DataFrame(list(score_lesoleil)).rename(columns = {"_id": "hash"})
    df_score_lesoleil["source"] = "lesoleil"
    df_score_lesoleil["point_view"] = df_score_lesoleil["view"]
    df_score_lesoleil["point_view5"] = df_score_lesoleil["view5"]
    df_score_lesoleil["point_view10"] = df_score_lesoleil["view10"] * 2
    df_score_lesoleil["point_view30"] = df_score_lesoleil["view30"] * 5
    df_score_lesoleil["point_view60"] = df_score_lesoleil["view60"] * 10
    return df_score_lesoleil

def get_df_score_lenouvelliste():
    db_online = connect_GLO7027()
    score_lenouvelliste_collection = db_online.get_collection("score_lenouvelliste")
    score_lenouvelliste = score_lenouvelliste_collection.find({})
    df_score_lenouvelliste = pd.DataFrame(list(score_lenouvelliste)).rename(columns = {"_id": "hash"})
    df_score_lenouvelliste["source"] = "lenouvelliste"
    df_score_lenouvelliste["point_view"] = df_score_lenouvelliste["view"]
    df_score_lenouvelliste["point_view5"] = df_score_lenouvelliste["view5"]
    df_score_lenouvelliste["point_view10"] = df_score_lenouvelliste["view10"] * 2
    df_score_lenouvelliste["point_view30"] = df_score_lenouvelliste["view30"] * 5
    df_score_lenouvelliste["point_view60"] = df_score_lenouvelliste["view60"] * 10
    
    return df_score_lenouvelliste


def get_df_score_lequotidien():
    db_online = connect_GLO7027()
    score_lequotidien_collection = db_online.get_collection("score_lequotidien")
    score_lequotidien = score_lequotidien_collection.find({})
    df_score_lequotidien = pd.DataFrame(list(score_lequotidien)).rename(columns = {"_id": "hash"})
    df_score_lequotidien["source"] = "lequotidien"
    df_score_lequotidien["point_view"] = df_score_lequotidien["view"]
    df_score_lequotidien["point_view5"] = df_score_lequotidien["view5"]
    df_score_lequotidien["point_view10"] = df_score_lequotidien["view10"] * 2
    df_score_lequotidien["point_view30"] = df_score_lequotidien["view30"] * 5
    df_score_lequotidien["point_view60"] = df_score_lequotidien["view60"] * 10
    return df_score_lequotidien

def get_df_score_latribune():
    db_online = connect_GLO7027()
    score_latribune_collection = db_online.get_collection("score_latribune")
    score_latribune = score_latribune_collection.find({})
    df_score_latribune = pd.DataFrame(list(score_latribune)).rename(columns = {"_id": "hash"})
    df_score_latribune["source"] = "latribune"
    df_score_latribune["point_view"] = df_score_latribune["view"]
    df_score_latribune["point_view5"] = df_score_latribune["view5"]
    df_score_latribune["point_view10"] = df_score_latribune["view10"] * 2
    df_score_latribune["point_view30"] = df_score_latribune["view30"] * 5
    df_score_latribune["point_view60"] = df_score_latribune["view60"] * 10
    return df_score_latribune

def get_df_score_lavoixdelest():
    db_online = connect_GLO7027()
    score_lavoixdelest_collection = db_online.get_collection("score_lavoixdelest")
    score_lavoixdelest = score_lavoixdelest_collection.find({})
    df_score_lavoixdelest = pd.DataFrame(list(score_lavoixdelest)).rename(columns = {"_id": "hash"})
    df_score_lavoixdelest["source"] = "lavoixdelest"
    df_score_lavoixdelest["point_view"] = df_score_lavoixdelest["view"]
    df_score_lavoixdelest["point_view5"] = df_score_lavoixdelest["view5"]
    df_score_lavoixdelest["point_view10"] = df_score_lavoixdelest["view10"] * 2
    df_score_lavoixdelest["point_view30"] = df_score_lavoixdelest["view30"] * 5
    df_score_lavoixdelest["point_view60"] = df_score_lavoixdelest["view60"] * 10
    return df_score_lavoixdelest

def merge_df_with_score_on_hash_key(df_with_info, df_score):
    return pd.merge(df_with_info, df_score, on="hash")
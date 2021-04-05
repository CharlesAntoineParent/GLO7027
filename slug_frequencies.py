import configparser
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections_utilis import get_train_publication_over_2019, merge_df_with_score_on_hash_key, get_df_score_ledroit, get_df_score_lesoleil, get_df_score_lenouvelliste, get_df_score_lequotidien, get_df_score_latribune, get_df_score_lavoixdelest

configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

def get_slug_1_with_hash_and_source():
    train_publication = get_train_publication_over_2019()
    df_slug_from_mongo = pd.json_normalize(list(train_publication), meta = ["id", "organizationKey"], record_path = ["publications"]).rename(columns = {"id": "hash", "slug.fr":"slug", "organizationKey" : "source"})
    df_slug = df_slug_from_mongo[["hash", "source", "slug"]]
    df_slug["slug_1"] = df_slug["slug"].apply(lambda x: x.split("/")[0] )
    df_slug = df_slug[["hash", "source", "slug_1"]]
    df_slug['slug_1']= np.where(df_slug['slug_1'] == "actualite",'actualites', df_slug['slug_1'])
    return df_slug

if __name__ == '__main__':

    #df_slug["slug.fr_2"] = df_slug["slug.fr"].apply(lambda x: "/".join(x.split("/")[:max(1, len(x.split("/"))-1)] ))


    df_slug = get_slug_1_with_hash_and_source()

    f, ax = plt.subplots(figsize=(7, 6))
    ax = df_slug["slug_1"].value_counts(normalize=True).head(6).plot(kind = "barh")
    plt.ylabel("Slug", fontsize=18)
    plt.xlabel("Fréquence", fontsize=18)
    ax.xaxis.grid(True)
    ax.set_yticklabels(list(df_slug["slug_1"].value_counts(normalize=True).head(6).keys()), rotation=0, fontsize=16)
    plt.show()



    df_slug_ledroit = df_slug[df_slug["source"] == "ledroit"]
    df_slug_lesoleil = df_slug[df_slug["source"] == "lesoleil"] 
    df_slug_lenouvelliste = df_slug[df_slug["source"] == "lenouvelliste"] 
    df_slug_lequotidien = df_slug[df_slug["source"] == "lequotidien"] 
    df_slug_latribune = df_slug[df_slug["source"] == "latribune"] 
    df_slug_lavoixdelest = df_slug[df_slug["source"] == "lavoixdelest"] 

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


    ####################################### Section score par source ######################################
    df_score_ledroit = get_df_score_ledroit()
    df_score_lesoleil = get_df_score_lesoleil()
    df_score_lenouvelliste = get_df_score_lenouvelliste()
    df_score_lequotidien = get_df_score_lequotidien()
    df_score_latribune = get_df_score_latribune()
    df_score_lavoixdelest = get_df_score_lavoixdelest()

    df_externalid_score_ledroit = merge_df_with_score_on_hash_key(df_slug_ledroit, df_score_ledroit)
    df_externalid_score_lesoleil = merge_df_with_score_on_hash_key(df_slug_lesoleil, df_score_lesoleil)
    df_externalid_score_lenouvelliste = merge_df_with_score_on_hash_key(df_slug_lenouvelliste, df_score_lenouvelliste)
    df_externalid_score_lequotidien = merge_df_with_score_on_hash_key(df_slug_lequotidien, df_score_lequotidien)
    df_externalid_score_latribune = merge_df_with_score_on_hash_key(df_slug_latribune, df_score_latribune)
    df_externalid_score_lavoixdelest = merge_df_with_score_on_hash_key(df_slug_lavoixdelest, df_score_lavoixdelest)
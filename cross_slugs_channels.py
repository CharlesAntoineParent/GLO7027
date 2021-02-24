from channel_frequencies import get_channel_with_hash
from slug_frequencies import get_slug_1_with_hash_and_source
import configparser
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


configParser = configparser.RawConfigParser()
configFilePath = r'config.txt'
configParser.read(configFilePath)
dataPath = configParser.get('config', 'dataPath')

if __name__ == '__main__':
    df_channel = get_channel_with_hash()
    df_slug_1 = get_slug_1_with_hash_and_source()

    df_slug_channel = pd.merge(df_slug_1, df_channel, on="hash")

    df_slug_channel["slug_cross_channels"] = df_slug_channel.slug_1 + "__" + df_slug_channel.channel

    f, ax = plt.subplots(figsize=(7, 6))
    ax = df_slug_channel["slug_cross_channels"].value_counts(normalize=True).head(10).plot(kind = "barh")
    #plt.title("Croisement Slug et Channel concatenés", fontsize=20)
    plt.xlabel("Fréquence",  fontsize=18)
    #plt.ylabel("")
    ax.xaxis.grid(True)
    ax.set_yticklabels(list(df_slug_channel["slug_cross_channels"].value_counts(normalize=True).head(10).keys()), rotation=0, fontsize=22)

    plt.show()




    df_slug_channel_ledroit = df_slug_channel[df_slug_channel["source"] == "ledroit"]
    df_slug_channel_lesoleil = df_slug_channel[df_slug_channel["source"] == "lesoleil"]
    df_slug_channel_lenouvelliste = df_slug_channel[df_slug_channel["source"] == "lenouvelliste"]
    df_slug_channel_lequotidien = df_slug_channel[df_slug_channel["source"] == "lequotidien"]
    df_slug_channel_latribune = df_slug_channel[df_slug_channel["source"] == "latribune"]
    df_slug_channel_lavoixdelest = df_slug_channel[df_slug_channel["source"] == "lavoixdelest"]

    fig, axes = plt.subplots(2, 3)
    plt.subplots_adjust(top=0.88, bottom=0.11, left=0.16, right=0.9, hspace=0.2, wspace=0.8)
    fig.suptitle("Fréquence slugs et channels combinés selon les sources", fontsize=16)

    df_slug_channel_ledroit["slug_cross_channels"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[0,0])
    axes[0,0].set_title("Le Droit")

    axes[0,0].set_ylabel("Slug_1 et channel combiné")
    df_slug_channel_lesoleil["slug_cross_channels"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[0,1])
    axes[0,1].set_title("Le Soleil")


    df_slug_channel_lenouvelliste["slug_cross_channels"].value_counts(normalize=True).head(10).plot(kind = "barh",ax = axes[0,2])
    axes[0,2].set_title("Le Nouvelliste")


    df_slug_channel_lequotidien["slug_cross_channels"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[1,0])
    axes[1,0].set_title("Le Quotidien")
    axes[1,0].set_xlabel("Fréquence")
    axes[1,0].set_ylabel("Slug_1 et channel combiné")
    df_slug_channel_latribune["slug_cross_channels"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[1,1])
    axes[1,1].set_title("La Tribune")
    axes[1,1].set_xlabel("Fréquence")

    df_slug_channel_lavoixdelest["slug_cross_channels"].value_counts(normalize=True).head(10).plot(kind = "barh", ax = axes[1,2])
    axes[1,2].set_title("La Voix de l'Est")
    axes[1,2].set_xlabel("Fréquence")


    plt.show()




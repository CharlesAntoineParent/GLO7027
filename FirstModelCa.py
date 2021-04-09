from collections_utilis import * 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from utils import *


df = getTrainingData()
trainArticle = get_train_article_over_2019()
df = pd.DataFrame(trainArticle)
df = df.drop(['type','templateName','canonicalUrlOverride', 'contents','visual', 'availableInPreview','lead','url','id'],axis=1)

df.keys()
chapterStructure = list()

for i in df['chapters'].values:
    structure = list()
    try:
        for j in i:
            structure.append(j['type'])
        chapterStructure.append(structure)
    except TypeError:
        chapterStructure.append(list())

df['chapters'] = chapterStructure

channelNettoye = list()
for value in df['channel'].values:
    try:
        channelNettoye.append(value['fr'])
    except KeyError:
        channelNettoye.append(value)

df['channel'] = channelNettoye

dichotomizedExternalId = list()
for i in df.externalIds.values:
    if i == {}:
        dichotomizedExternalId.append(0)
    else:
        dichotomizedExternalId.append(1)

df['externalIds'] = dichotomizedExternalId

def day_of_week_num(dts):
    return (dts.astype('datetime64[D]').view('int64') - 4) % 7

df['dayOfWeek'] = [day_of_week_num(i) for i in df['creationDate'].values]
df = df.drop(['creationDate','modificationDate'],axis=1)
auteurNettoye = list()
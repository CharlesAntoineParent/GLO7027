from utils import *
import pandas as pd
import datetime
from tqdm import tqdm
from time import sleep
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np


for i in data.keys():
    for j in data[i].keys():
        if len(data[i][j]['author']) > 1:
            print(j)



with open('data/article2019.pickle', 'rb') as handle:
    data = pickle.load(handle)


## On va d'abord regarder la liste des auteurs

ListOfAuthors = list()

for journal in data.keys():
    for article in data[journal].values():
       
        try:
            ListOfAuthors.append(article['author'])
        except KeyError:
            continue 

ListOfAuthors = list(set(ListOfAuthors))

len(ListOfAuthors)

## Jusqu'à maintenant on à 5249 auteurs, on peut voir beaucoup d'erreur, la première qu'on va viser à éliminer, est les nom qui commencent par des espaces. ex : " Jean-Thomas Léveillé" deviendra "Jean-Thomas Léveillé"

ListOfAuthors = list(set([author.lstrip(' ') for author in ListOfAuthors]))

len(ListOfAuthors)

## Maintenant on est rendu avec 5246, la prochaine correction vise à remplacer les auteurs qui sont différents seulement avec les lettres majuscules ex:La Presse Canadienne', 'La Presse et La Presse canadienne

ListOfAuthors = list(set([author.lower() for author in ListOfAuthors]))

ListOfAuthors.sort()
len(ListOfAuthors)

## Maintenant on est rendu avec 4988 auteurs, la prochaine correction vise à remplacer les articles qui ont deux auteurs on va donc séparer les ' et '
#
NewListOfAuthors = list()
for auteur in ListOfAuthors:
    if auteur.find(' et ') > -1:
        auteur = auteur.split(' et ')
        NewListOfAuthors.append(auteur[0])
        NewListOfAuthors.append(auteur[1])
    else:
        NewListOfAuthors.append(auteur)

ListOfAuthors = list(set(NewListOfAuthors))
len(ListOfAuthors)


## On  est rendu à 4767 données, maintenant, on peut voir que dans beaucoup de noms, les espaces sont remplacées par cette hexa \xa0. On va donc le remplacer

ListOfAuthors = list(set([author.replace("\xa0", " ") for author in ListOfAuthors]))
ListOfAuthors.sort()
len(ListOfAuthors)



# Maintenant on est rendu avec 4742 auteurs, la prochaine correction vise à remplacer les articles qui ont deux auteurs on va donc séparer les ', '


NewListOfAuthors = list()
for auteur in ListOfAuthors:
    if auteur.find(', ') > -1:
        auteur = auteur.split(', ')
        NewListOfAuthors.append(auteur[0])
        NewListOfAuthors.append(auteur[1])
    else:
        NewListOfAuthors.append(auteur)

ListOfAuthors = list(set(NewListOfAuthors))
ListOfAuthors.sort()
len(ListOfAuthors)


# Maintenant on est rendu avec 4645 auteurs, la prochaine correction vise à remplacer les charactères spéciaux du français Ex: é,è etc


NewListOfAuthors = list()
for auteur in ListOfAuthors:
    auteur = auteur.replace('î','i')
    auteur = auteur.replace('è','e')
    auteur = auteur.replace('é','e')
    auteur = auteur.replace('ô','o')
    auteur = auteur.replace('#','')
    auteur = auteur.replace('*','')
    NewListOfAuthors.append(auteur)

ListOfAuthors = list(set(NewListOfAuthors))
ListOfAuthors.sort()
len(ListOfAuthors)


# Maintenant on est rendu avec 4645 auteurs, la prochaine correction vise à remplacer les parenthèses et leur contenues


NewListOfAuthors = list()
for auteur in ListOfAuthors:

    NewListOfAuthors.append(re.sub(r"\s*\(.*\)\s*","",auteur))

ListOfAuthors = list(set(NewListOfAuthors))
ListOfAuthors.sort()
len(ListOfAuthors)

# Maintenant on est rendu avec 4645 auteurs, prochaine correction vise à split à  ' - ' et enlever la dernière partie. En général c'est le poste de la personne

NewListOfAuthors = list()
for auteur in ListOfAuthors:
    if auteur.find(' - ') > -1:

        NewListOfAuthors.append(auteur.split(' - ')[0])
    else:
        NewListOfAuthors.append(auteur)

ListOfAuthors = list(set(NewListOfAuthors))
ListOfAuthors.sort()
len(ListOfAuthors)


# Maintenant on est rendu avec 4645 auteurs, prochaine correction vise à split à  '\r\n' et enlever la dernière partie. En général c'est le poste de la personne

NewListOfAuthors = list()
for auteur in ListOfAuthors:
    if auteur.find('\r\n') > -1:
        auteur = auteur.split('\r\n')
        NewListOfAuthors.append(auteur[0])
        NewListOfAuthors.append(auteur[1])
    else:
        NewListOfAuthors.append(auteur)
ListOfAuthors = list(set(NewListOfAuthors))
ListOfAuthors.sort()
len(ListOfAuthors)





NewListOfAuthors = list()
for auteur in ListOfAuthors:

    auteur = re.sub(r" .*@.*","",auteur)
    if len(auteur) > 5 :
        NewListOfAuthors.append(auteur.lstrip(' '))

ListOfAuthors = list(set(NewListOfAuthors))
ListOfAuthors.sort()
len(ListOfAuthors)


for auteur1 in ListOfAuthors:
    for auteur2 in ListOfAuthors:
        if auteur1 != auteur2:
            if nltk.edit_distance(auteur1, auteur2) < 2:
                print(auteur1, auteur2)

#

## Maintenant on applique la logique au auteur dans data
##D'abord on va créer une liste d'auteur par article

for journal in data.keys():
    
    for article in list(data[journal].keys()):

        try:
            
            auteur = [data[journal][article]['author']]
            auteur = [i.lower() for i in auteur] # On enleve les lettres majuscules
            
            new_auteur = list()
            for i in auteur:
                if i.find(' et ') > -1:

                    new_auteur.append(i.split(' et ')[0])
                    new_auteur.append(i.split(' et ')[1])

                elif i.find(', ') > -1:
                    new_auteur.append(i.split(', ')[0])
                    new_auteur.append(i.split(', ')[1])

                elif i.find('\r\n') > -1:
                    new_auteur.append(i.split('\r\n')[0])
                    new_auteur.append(i.split('\r\n')[1])                    


                else:
                    new_auteur.append(i)

            
            auteur = new_auteur

            auteur = [i.replace("\xa0", " ") for i in auteur]

            auteur = [i.replace("é", "e") for i in auteur]
            auteur = [i.replace("è", "e") for i in auteur]
            auteur = [i.replace("î", "i") for i in auteur]
            auteur = [i.replace("ô", "o") for i in auteur]
            auteur = [i.replace("#", "") for i in auteur]
            auteur = [i.replace("*", "") for i in auteur]
            auteur = [i.replace(".-", "-") for i in auteur]
            
            auteur = [re.sub(r"\s*\(.*\)\s*","",i) for i in auteur]
            auteur = [re.sub(r" .*@.*","",i) for i in auteur]
            auteur = [i for i in auteur if len(i) > 5]
            data[journal][article]['author'] = auteur
            
                   


        except KeyError:
            data[journal][article]['author'] = list()        



list(data['lesoleil'].values())[0]


journalisteDict = dict()


for journal in data.keys():
    for article in data[journal].keys():
        try:
            if len(data[journal][article]['author']) != 0 :
                for autor in data[journal][article]['author']:
                    print(autor)
                    
                    try:
                        
                        journalisteDict[autor]['count'] += 1
                        journalisteDict[autor]['View'] += data[journal][article]['View']
                        journalisteDict[autor]['View5'] += data[journal][article]['View5']
                        journalisteDict[autor]['View10'] += data[journal][article]['View10']
                        journalisteDict[autor]['View30'] += data[journal][article]['View30']
                        journalisteDict[autor]['View60'] += data[journal][article]['View60']

                    except KeyError:
                        journalisteDict[autor] = dict()
                        journalisteDict[autor]['count'] = 1
                        journalisteDict[autor]['View'] = data[journal][article]['View']
                        journalisteDict[autor]['View5'] = data[journal][article]['View5']
                        journalisteDict[autor]['View10'] = data[journal][article]['View10']
                        journalisteDict[autor]['View30'] = data[journal][article]['View30']
                        journalisteDict[autor]['View60'] = data[journal][article]['View60']
        except KeyError:
            pass


maxCount = 1
journaliste = ''
for journal,view in journalisteDict.items():
    if 'la presse canadienne' != journal:
        if view['count'] > maxCount:
            journaliste = journal
            maxCount = view['count']

journalisteDict


test = sorted(journalisteDict.items(), key=lambda x: x[1]['count'], reverse=True)
plt.hist([view['count'] for journal,view in journalisteDict.items()],40,(0,300))

plt.ylabel("Nombre d'auteurs")
plt.xlabel("Quantité d'article par auteur")   
plt.show()


df = pd.DataFrame.from_dict(journalisteDict,orient='index')

df['scoreMoyen'] = (df['View'] + df['View5'] + 2 *df['View10'] +  5 * df['View30'] + 10 *df['View60']) / df['count']
Under5 = df[df['count'] <= 5 ]
Between5And200 = df[df['count'] > 5][df[df['count'] > 5]['count']<=200]
MoreThen200 = df[df['count'] > 200 ]

boxplotData = [list(Under5['scoreMoyen'].values), list(Between5And200['scoreMoyen'].values), list(MoreThen200['scoreMoyen'].values)]
label = ['Under 5 ', 'Between 5 and 200', 'more than 200']

Under5.groupby().sum()
sum(Under5['scoreMoyen'].values)

fig = plt.figure(figsize =(10, 7)) 
ax = fig.add_subplot(111)
ax.boxplot(boxplotData)
ax.set_xticklabels(label)
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
viewLabel = ['View',  'View5',  'View10',  'View30',  'View60']
ax.pie([sum(Between5And200['View'].values),sum(Between5And200['View5'].values),sum(Between5And200['View10'].values),sum(Between5And200['View30'].values),sum(Between5And200['View60'].values)], labels = viewLabel,autopct='%1.2f%%')
ax.set_title('Repartion des types de vues de journalistes avec moins de 5 aricles')
plt.show()


for journal in data.keys():
    for article in data[journal].keys():
        df = data[journal][article]
        data[journal][article]['score'] = df['View'] + df['View5'] + 2 *df['View10'] +  5 * df['View30'] + 10 *df['View60']

myleneMoisanScore = list()
OtherScore = list()
jocelyneRicherScore = list()


for journal in data.keys():
    for article in data[journal].keys():
        author =  data[journal][article]['author']
        score =  data[journal][article]['score']
        if "jocelyne richer" in author:
            jocelyneRicherScore.append(score)

        if "mylene moisan" in author:
            myleneMoisanScore.append(score)
        else:
            OtherScore.append(score)

myleneMoisanScore.sort()
OtherScore.sort()
jocelyneRicherScore.sort()
myleneMoisanScore = myleneMoisanScore[round(0.1*len(myleneMoisanScore)):round(0.9*len(myleneMoisanScore))]
OtherScore = OtherScore[round(0.1*len(OtherScore)):round(0.9*len(OtherScore))]
jocelyneRicherScore = jocelyneRicherScore[round(0.1*len(jocelyneRicherScore)):round(0.9*len(jocelyneRicherScore))]



data = [myleneMoisanScore,OtherScore,jocelyneRicherScore]
label = ["Mylene Moisan", "Autres journalistes", "Jocelyne Richer"]
plt.figure(figsize=(8,6))
plt.hist(data, bins=list(range(0,400,40)), density=True, histtype='bar', color=colors, label=label)
plt.xlabel("Score des articles", size=14)
plt.ylabel("Fréquence représentant le nombre d'articles", size=14)
plt.title("Distribution du score des articles pour différents auteurs")
plt.legend(loc='upper right')
plt.show()
0xbe609eacbfca10f6e5504d39e3b113f808389056


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

colors = ['red', 'tan', 'lime']
ax0.hist(data, bins=list(range(0,2000,100)), density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

if __name__ == '__main__':

    journaux = [folder for folder in os.listdir('data/analytics') if not folder.startswith('.')]

    allJournauxSummary = dict()

    for journal in journaux:

        allJournauxSummary[journal] = journalSummary(journal,10)
        sleepUpdate = 1 / len(allJournauxSummary[journal].keys())
        

        for articleHash in tqdm(list(allJournauxSummary[journal].keys())[0:1000]):
            sleep(sleepUpdate)
            try:
                article = getArticle(articleHash)
            except FileNotFoundError:
                continue
            try:
                author = article[1]['authors'][0]['name']
                allJournauxSummary[journal][articleHash]['author'] = author
                
            except (IndexError,KeyError):

                print(articleHash)
                continue
        

    

    with open('articleByAuthor.pickle', 'wb') as handle:
        pickle.dump(allJournauxSummary, handle, protocol=pickle.HIGHEST_PROTOCOL)


auteur = list()
for journal in allJournauxSummary.keys():

    for article in allJournauxSummary[journal].keys():
        try:
            auteur.append(allJournauxSummary[journal][article]['author'])
        except KeyError:
            pass

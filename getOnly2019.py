import utils
import pandas as pd
import pickle


with open('data/articleByAuthor.pickle', 'rb') as handle:
    data = pickle.load(handle)

list(data.keys())[0]
type(data['lavoixdelest'])
list(data['lavoixdelest'])[0]
a = utils.getArticle('6725af013e44f374ec001b19e236ecac')

repartitionSelonAnnee = dict()
articleAvecAnneeEtScore = dict()

for journal in data.keys():
    for article in list(data[journal].keys()):
        
        try:
            currentArticle = utils.getArticle(article)
            publicationYear = currentArticle[0][0]['publications'][0]['publicationDate'][0:4]
            score = data[journal][article]['View'] + data[journal][article]['View5'] + 2 *data[journal][article]['View10'] +  5 * data[journal][article]['View30'] + 10 *data[journal][article]['View60']

            try:
                repartitionSelonAnnee[publicationYear]["NbArticle"] += 1
                repartitionSelonAnnee[publicationYear]["Score"] += score


            except KeyError:
                repartitionSelonAnnee[publicationYear] = {"NbArticle":0, "Score": 0 }
                repartitionSelonAnnee[publicationYear]["NbArticle"] += 1
                repartitionSelonAnnee[publicationYear]["Score"] += score


            try:
                articleAvecAnneeEtScore[article]["NbArticle"] += 1
                articleAvecAnneeEtScore[article]["Score"] += score
                

            except KeyError:
                articleAvecAnneeEtScore[article] = {"Annee":0, "Score": 0 }
                articleAvecAnneeEtScore[article]["Annee"] = publicationYear
                articleAvecAnneeEtScore[article]["Score"] += score


        except FileNotFoundError:
            pass


Result.groupby('')


Result = pd.DataFrame.from_dict(articleAvecAnneeEtScore,orient='index')
Result.groupby('Annee').agg({'Score': ['mean','std', 'min', 'max','median','sum']})

Result.to_csv("DonneesJournauxParAnnee.csv")


Result = Result[Result["Annee"] == "2019"]

Article2019 = set(Result.index)
data2019 = dict()

for journal in data.keys():
    for articleHash in data[journal].keys():
        if articleHash in Article2019:
            try:            
                data2019[journal][articleHash] = data[journal][articleHash]
            except KeyError:
                data2019[journal] = dict()
                data2019[journal][articleHash] = data[journal][articleHash]



with open('data/article2019.pickle', 'wb') as handle:
    pickle.dump(data2019, handle, protocol=pickle.HIGHEST_PROTOCOL)

             

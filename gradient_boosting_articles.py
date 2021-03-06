from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
import shap
import FeaturesEvaluator
import numpy as np

with open('training_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

data = data.dropna().reset_index()

data = data.drop(['_id', 'count', 'organizationKey',
                'view', 'view5','view10', 'view30','view60',
                'point_view', 'point_view5', 'point_view10', 'point_view30', 'point_view60'], axis=1)


def get_list(string):
    try:
        return [string]
    except IndexError:
        return ["vide"]

data.publications = data.publications.apply(lambda x : get_list(x))

data.keys()

data_ledroit =   data[data.source == "ledroit"]
#data_lesoleil =   data[data.source == "lesoleil"]
#data_lenouvelliste =   data[data.source == "lenouvelliste"]
#data_lequotidien =   data[data.source == "lequotidien"]
#data_latribune =   data[data.source == "latribune"]
#data_lavoixdelest =   data[data.source == "lavoixdelest"]


#data_ledroit = data_ledroit.reset_index()
#data_lesoleil = data_lesoleil.reset_index()
#data_lenouvelliste = data_lenouvelliste.reset_index()
#data_lequotidien = data_lequotidien.reset_index()
#data_latribune = data_latribune.reset_index()
#data_lavoixdelest = data_lavoixdelest.reset_index()

def get_first_title(list):
    try:
        return [list[0]]
    except IndexError:
        return ["vide"]




def get_formatted_training_data(df):
    #data.title = data.title.apply(lambda x : get_first_title(x))

    mlb = MultiLabelBinarizer()
    title = pd.DataFrame(mlb.fit_transform(df.title),columns=mlb.classes_, index=df.index)
    title_only_alpha = title.loc[:, [k.isalpha() for k in title.keys()]]
    title_only_alpha_frequent = title_only_alpha.loc[:, title_only_alpha.sum(axis=0) > 50]
    df = pd.concat([df.drop('title', axis = 1), title_only_alpha_frequent.add_prefix('title_')], axis=1)

    authors = pd.DataFrame(mlb.fit_transform(df.authors),columns=mlb.classes_, index=df.index)
    authors_only_alpha = authors.loc[:, [k.isalpha() for k in authors.keys()]]
    authors_only_alpha_frequent = authors_only_alpha.loc[:, authors_only_alpha.sum(axis=0) > 10]
    df = pd.concat([df.drop('authors', axis = 1), authors_only_alpha_frequent.add_prefix('authors_')], axis=1)

    channel = pd.DataFrame(mlb.fit_transform(df.channel),columns=mlb.classes_, index=df.index)
    channel_only_alpha = channel.loc[:, [k.isalpha() for k in channel.keys()]]
    channel_only_alpha_frequent = channel_only_alpha.loc[:, channel_only_alpha.sum(axis=0) > 10]
    df = pd.concat([df.drop('channel', axis = 1), channel_only_alpha_frequent.add_prefix('channel_')], axis=1)

    slug = pd.DataFrame(mlb.fit_transform(df.publications),columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop('publications', axis = 1), slug.add_prefix('slug_')], axis=1)

    df.reset_index(inplace=True)

    df_count_chapters = pd.json_normalize(df.chapters.apply(lambda x : Counter(x))).fillna(0)
    df_count_chapters = df_count_chapters.add_prefix('chapters_count_').reset_index(inplace=True)
    df = pd.concat([df.drop('chapters', axis = 1), df_count_chapters], axis=1)
    return df



df_ledroit = get_formatted_training_data(data_ledroit)
#df_lesoleil = get_formatted_training_data(data_lesoleil)
#df_lenouvelliste = get_formatted_training_data(data_lenouvelliste)
#df_lequotidien = get_formatted_training_data(data_lequotidien)
#df_latribune = get_formatted_training_data(data_latribune)
#df_lavoixdelest = get_formatted_training_data(data_lavoixdelest)


df_ledroit["random_noise"] = np.random.normal(loc=0, scale=1, size = df_ledroit.shape[0])
#df_lesoleil = get_formatted_training_data(data_lesoleil)
#df_lenouvelliste = get_formatted_training_data(data_lenouvelliste)
#df_lequotidien = get_formatted_training_data(data_lequotidien)
#df_latribune = get_formatted_training_data(data_latribune)
#df_lavoixdelest = get_formatted_training_data(data_lavoixdelest)

Y_ledroit = df_ledroit["score"]
#Y_lesoleil = df_lesoleil["score"]
#Y_lenouvelliste = df_lenouvelliste["score"]
#Y_lequotidien = df_lequotidien["score"]
#Y_latribune = df_latribune["score"]
#Y_lavoixdelest = df_lavoixdelest["score"]


X_ledroit = df_ledroit.drop(["score", "source", "level_0", "index"], axis = 1)
#X_lesoleil = df_lesoleil.drop(["score", "source", "level_0", "index"], axis = 1)
#X_lenouvelliste = df_lenouvelliste.drop(["score", "source", "level_0", "index"], axis = 1)
#X_lequotidien = df_lequotidien.drop(["score", "source", "level_0", "index"], axis = 1)
#X_latribune = df_latribune.drop(["score", "source", "level_0", "index"], axis = 1)
#X_lavoixdelest = df_lavoixdelest.drop(["score", "source", "level_0", "index"], axis = 1)

#[str(i) for i in X.title[3]] 
#X.title = X.title.apply(lambda x : map(str, x))
#X.dropna()




X_train_ledroit, X_test_ledroit, y_train_ledroit, y_test_ledroit = train_test_split(X_ledroit, Y_ledroit, test_size=0.1, random_state=42)
#X_train_lesoleil, X_test_lesoleil, y_train_lesoleil, y_test_lesoleil = train_test_split(X_lesoleil, Y_lesoleil, test_size=0.1, random_state=42)
#X_train_lenouvelliste, X_test_lenouvelliste, y_train_lenouvelliste, y_test_lenouvelliste = train_test_split(X_lenouvelliste, Y_lenouvelliste, test_size=0.1, random_state=42)
#X_train_lequotidien, X_test_lequotidien, y_train_lequotidien, y_test_lequotidien = train_test_split(X_lequotidien, Y_lequotidien, test_size=0.1, random_state=42)
#X_train_latribune, X_test_latribune, y_train_latribune, y_test_latribune = train_test_split(X_latribune, Y_latribune, test_size=0.1, random_state=42)
#X_train_lavoixdelest, X_test_lavoixdelest, y_train_lavoixdelest, y_test_lavoixdelest = train_test_split(X_lavoixdelest, Y_lavoixdelest, test_size=0.1, random_state=42)

train_hash_ledroit = X_train_ledroit['hash']  
#train_hash_lesoleil = X_train_lesoleil['hash'] 
#train_hash_lenouvelliste = X_train_lenouvelliste['hash'] 
#train_hash_lequotidien = X_train_lequotidien['hash'] 
#train_hash_latribune = X_train_latribune['hash'] 
#train_hash_lavoixdelest = X_train_lavoixdelest['hash'] 

test_hash_ledroit = X_test_ledroit['hash']
#test_hash_lesoleil = X_test_lesoleil['hash']
#test_hash_lenouvelliste = X_test_lenouvelliste['hash']
#test_hash_lequotidien = X_test_lequotidien['hash']
#test_hash_latribune = X_test_latribune['hash']
#test_hash_lavoixdelest = X_test_lavoixdelest['hash']


X_train_ledroit = X_train_ledroit.drop("hash", axis = 1)
#X_train_lesoleil = X_train_lesoleil.drop("hash", axis = 1)
#X_train_lenouvelliste = X_train_lenouvelliste.drop("hash", axis = 1)
#X_train_lequotidien = X_train_lequotidien.drop("hash", axis = 1)
#X_train_latribune = X_train_latribune.drop("hash", axis = 1)
#X_train_lavoixdelest = X_train_lavoixdelest.drop("hash", axis = 1)

X_test_ledroit = X_test_ledroit.drop("hash", axis = 1)
#X_test_lesoleil = X_test_lesoleil.drop("hash", axis = 1)
#X_test_lenouvelliste = X_test_lenouvelliste.drop("hash", axis = 1)
#X_test_lequotidien = X_test_lequotidien.drop("hash", axis = 1)
#X_test_latribune = X_test_latribune.drop("hash", axis = 1)
#X_test_lavoixdelest = X_test_lavoixdelest.drop("hash", axis = 1)


params = {'n_estimators': 128,
        'max_depth': 7,
        'min_samples_split': 5,
        'learning_rate': 0.01,
        'loss': 'ls'}

reg_ledroit = ensemble.GradientBoostingRegressor(**params)
#reg_lesoleil = ensemble.GradientBoostingRegressor(**params)
#reg_lenouvelliste = ensemble.GradientBoostingRegressor(**params)
#reg_lequotidien = ensemble.GradientBoostingRegressor(**params)
#reg_latribune = ensemble.GradientBoostingRegressor(**params)
#reg_lavoixdelest = ensemble.GradientBoostingRegressor(**params)

reg_ledroit.fit(X_train_ledroit, y_train_ledroit)
#reg_lesoleil.fit(X_train_lesoleil, y_train_lesoleil)
#reg_lenouvelliste.fit(X_train_lenouvelliste, y_train_lenouvelliste)
#reg_lequotidien.fit(X_train_lequotidien, y_train_lequotidien)
#reg_latribune.fit(X_train_latribune, y_train_latribune)
#reg_lavoixdelest.fit(X_train_lavoixdelest, y_train_lavoixdelest)

predict_ledroit = reg_ledroit.predict(X_test_ledroit)
#predict_lesoleil = reg_lesoleil.predict(X_test_lesoleil)
#predict_lenouvelliste = reg_lenouvelliste.predict(X_test_lenouvelliste)
#predict_lequotidien = reg_lequotidien.predict(X_test_lequotidien)
#predict_latribune = reg_latribune.predict(X_test_latribune)
#predict_lavoixdelest = reg_lavoixdelest.predict(X_test_lavoixdelest)


reg_ledroit.score(X_test_ledroit,y_test_ledroit )





ledroit_Evaluator = FeaturesEvaluator.FeaturesEvaluator(reg_ledroit, X_train_ledroit, y_train_ledroit, X_test_ledroit, y_test_ledroit)





r2 = ledroit_Evaluator.get_r2_score()
pd_importance = pd.DataFrame.from_dict(ledroit_Evaluator.get_sorted_importance(False))
pd_permutation_importance_val = pd.DataFrame.from_dict(ledroit_Evaluator.get_sorted_permutaion_importance(False))
pd_SHAP = pd.DataFrame.from_dict(ledroit_Evaluator.get_sorted_SHAP_values())
treshhold = pd_SHAP[pd_SHAP["feature"] == "random_noise"]
treshhold = float(abs(treshhold.SHAP_value))

dfs = [pd_importance, pd_permutation_importance_val, pd_SHAP]
dfs[0].join(dfs[1:])

test = pd_SHAP[pd_SHAP.SHAP_value < treshhold]
test = test[test.SHAP_value > -treshhold]




a = ledroit_Evaluator.get_sorted_SHAP_values()

#ledroit_Evaluator.get_correlation_heatmap()
pd.concat(
    (iDF.set_index('feature') for iDF in [pd_importance, pd_permutation_importance_val, pd_SHAP]),
    axis=1, join='inner'
)

df_total = pd_importance.merge(pd_permutation_importance_val, how='left').merge(pd_SHAP, how='left')



ledroit_Evaluator.get_SHAP_values_bar_plot()
ledroit_Evaluator.get_SHAP_values_impact_model_output()
#ledroit_Evaluator.get_dependence_plot(["rank(1)", "monde"])



reg_ledroit_clean = ensemble.GradientBoostingRegressor(**params)

X_train_ledroit_clean = X_train_ledroit.drop(test.feature,  axis = 1)
reg_ledroit_clean.fit(X_train_ledroit_clean, y_train_ledroit)


X_test_ledroit_clean  = X_test_ledroit.drop(test.feature,  axis = 1)


reg_ledroit_clean.score(X_test_ledroit_clean,y_test_ledroit )





ledroit_Evaluator = FeaturesEvaluator.FeaturesEvaluator(reg_ledroit, X_train_ledroit, y_train_ledroit, X_test_ledroit, y_test_ledroit)








params = {'n_estimators': 128,
        'max_depth': 7,
        'min_samples_split': 5,
        'learning_rate': 0.01,
        'loss': 'ls'}

reg_ledroit = ensemble.GradientBoostingRegressor(**params)

selector = RFECV(reg_ledroit, step=1, cv=5, verbose =3 , min_features_to_select=30)
selector = selector.fit(X_train_ledroit, y_train_ledroit)

selector.support_
selector.grid_scores_
X_to_remove = X_train_ledroit.keys()[np.logical_not(selector.support_)]

reg_ledroit.fit(X_train_ledroit, y_train_ledroit)


predict_ledroit = reg_ledroit.predict(X_test_ledroit)

test = pd.DataFrame(selector.transform(X_train_ledroit))
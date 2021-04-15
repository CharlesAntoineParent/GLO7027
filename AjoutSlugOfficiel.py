import pandas as pd



dfSlug = pd.read_csv('listeSlug.txt',sep='\t')
df = pd.read_pickle('test.pkl')

df = pd.merge(df,dfSlug,how='outer',left_on="_id",right_on="article")



dfSlug['categorie'].value_counts()
import pandas as pd
import pickle

df = pd.read_csv('data\hateval2019_en_test.csv')
data = []
for index, row in df.iterrows():
    if row['HS'] == 0:
        label = 'none'
    else:
        label = 'hate'
    data.append({'id': row['id'], 'text': row['text'], 'label': label})

df = pd.read_csv('data\hateval2019_en_train.csv')
for index, row in df.iterrows():
    if row['HS'] == 0:
        label = 'none'
    else:
        label = 'hate'
    data.append({'id': row['id'], 'text': row['text'], 'label': label})
pickle.dump(data, open('data\sem_eval.pkl', "wb"))
df1 = pickle.load(open('data\sem_eval.pkl', 'rb'))

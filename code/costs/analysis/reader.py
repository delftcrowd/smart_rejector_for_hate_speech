import csv
from pickle import TRUE
from pydoc import doc
import pandas as pd

PATH = "F:\Thesis\Experiments\Costs\Results\/03-06-2022 (FEEDBACK) ME-100.csv"

data = pd.read_csv(PATH)
df = pd.DataFrame()
TYPES = ["TP", "TN", "FP", "FN", "REJ"]

for index, row in data.iterrows():
    r = {}
    for type in TYPES:
        for i in range(1, 5):
            decision = row.filter(regex=f"^ME{type}{i}s\.").values[0]
            value = row.filter(regex=f"^ME{type}{i}v\.").values[0]
            me = None

            if decision == 'Agree':
                me = value
            elif decision == 'Disagree':
                me = -value
            elif decision == 'Neutral':
                me = 0

            r[f"ME{type}{i}"] = me

    df = df.append(r, ignore_index=True)

print(df)

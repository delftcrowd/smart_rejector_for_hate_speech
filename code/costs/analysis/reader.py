import csv
from pickle import TRUE
from pydoc import doc
import pandas as pd

PATH = "F:\Thesis\Experiments\Costs\Results\/03-06-2022 (FEEDBACK) ME-100.csv"
TYPES = ["TP", "TN", "FP", "FN", "REJ"]

data = pd.read_csv(PATH)


def get_value(row, scale, type, index, question):
    return row.filter(regex=f"^{scale}{type}{index}{question}\.").values[0]


def convert_me(decision, value):
    if decision == 'Agree':
        return value
    elif decision == 'Disagree':
        return -value
    elif decision == 'Neutral':
        return 0


def magnitude_estimates(data):
    df = pd.DataFrame()

    for _index, row in data.iterrows():
        r = {}
        for type in TYPES:
            for i in range(1, 5):
                decision = get_value(row, "ME", type, i, "s")
                value = get_value(row, "ME", type, i, "v")
                me = convert_me(decision, value)
                r[f"ME{type}{i}"] = me

        df = df.append(r, ignore_index=True)

    return df


print(magnitude_estimates(data))

import csv
from pickle import TRUE
from pydoc import doc
from numpy import mean
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


def normalize(data, magnitude_estimates):
    new_df = pd.DataFrame()

    for index, row in data.iterrows():
        mes = magnitude_estimates.iloc[[index]]
        pivot = pivot_value(row)
        normalized_mes = mes.div(pivot)
        new_df = new_df.append(normalized_mes, ignore_index=True)

    return new_df


def pivot_value(row):
    NAME = "G20Q51"
    str_dis = row.filter(regex=f"^{NAME}\[SQ001\]\.").values[0]
    som_dis = row.filter(regex=f"^{NAME}\[SQ002\]\.").values[0]
    dis = row.filter(regex=f"^{NAME}\[SQ003\]\.").values[0]
    som_agr = row.filter(regex=f"^{NAME}\[SQ005\]\.").values[0]
    agr = row.filter(regex=f"^{NAME}\[SQ006\]\.").values[0]
    str_agr = row.filter(regex=f"^{NAME}\[SQ007\]\.").values[0]
    calibration_vals = [str_dis, som_dis, dis, som_agr, agr, str_agr]
    absolute_cal_vals = [abs(val) for val in calibration_vals]
    return mean(absolute_cal_vals)


mes = magnitude_estimates(data)
print(mes)
normalized_mes = normalize(data, mes)
print(normalized_mes)

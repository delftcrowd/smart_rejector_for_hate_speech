import numpy as np
import pandas as pd
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

PATH = "F:\Thesis\Experiments\Costs\Results\/03-06-2022 (FEEDBACK) ME-100.csv"
TYPES = ["TP", "TN", "FP", "FN", "REJ"]
SCALES = ["S100", "ME"]


def get_value(row, scale, type, index, question):
    return row.filter(regex=f"^{scale}{type}{index}{question}\.").values[0]


def filter_row(row, scale, type):
    return row.filter(regex=f"{scale}{type}")


def convert_100(decision, agree_value, disagree_value):
    if decision == 'Agree':
        return agree_value
    elif decision == 'Disagree':
        return -disagree_value
    elif decision == 'Neutral':
        return 0


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


def s100_values(data):
    df = pd.DataFrame()

    for _index, row in data.iterrows():
        r = {}
        for type in TYPES:
            for i in range(1, 5):
                decision = get_value(row, "S100", type, i, "s")
                agree_value = get_value(row, "S100", type, i, "a\[SQ001\]")
                disagree_value = get_value(row, "S100", type, i, "d\[SQ001\]")
                v100 = convert_100(decision, agree_value, disagree_value)
                r[f"S100{type}{i}"] = v100

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
    return np.mean(absolute_cal_vals)


def calculate_mean(data, scale, type):
    type_values = data.filter(regex=f"{scale}{type}", axis=1)
    column_means = type_values.mean()
    return column_means.mean()


def convert_data(data):
    mes = magnitude_estimates(data)
    normalized_mes = normalize(data, mes)
    s100 = s100_values(data)
    return pd.concat([normalized_mes, s100], axis=1)


def reliability(data, scale='', type=''):
    if scale != '' or type != '':
        data = data.filter(regex=f"{scale}{type}", axis=1)

    data = data.values.tolist()
    # TODO: remove this for real data
    data = [data[0], data[0]]

    try:
        alpha = krippendorff.alpha(reliability_data=data)
    except:
        # The value domain should consist of at least one item
        # If not, then catch exception and retun np.nan
        alpha = np.nan

    return alpha


def print_means(data):
    print("===================")
    for scale in SCALES:
        print(f"{scale} scale")
        for type in TYPES:
            print(type, calculate_mean(data, "ME", type))
        print("===================")


def print_reliabilities(data):
    print("===================")
    print("alpha complete data:", reliability(data))
    print("===================")
    for scale in SCALES:
        print(f"{scale} scale")
        for type in TYPES:
            print(type, reliability(data, scale=scale, type=type))
        print("===================")


def plot_validity(data):
    mes = data.filter(regex="ME", axis=1)
    s100 = data.filter(regex="S100", axis=1)
    mes = mes.mean().tolist()
    s100 = s100.mean().tolist()

    sns.regplot(x=mes, y=s100)
    plt.title("Correlation")
    plt.xlabel("ME")
    plt.ylabel("100-level")
    plt.show()


def print_correlation(data):
    mes = data.filter(regex="ME", axis=1)
    s100 = data.filter(regex="S100", axis=1)
    mes = mes.mean().tolist()
    s100 = s100.mean().tolist()

    print("Pearson", stats.pearsonr(mes, s100))
    print("Spearman", stats.spearmanr(mes, s100))
    print("Kendall", stats.kendalltau(mes, s100))


data = pd.read_csv(PATH)
data = convert_data(data)
print(data)
print_means(data)
print_reliabilities(data)
plot_validity(data)
print_correlation(data)

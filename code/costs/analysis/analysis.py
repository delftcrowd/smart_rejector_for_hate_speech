import numpy as np
import pandas as pd
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

TYPES = ["TP", "TN", "FP", "FN", "REJ"]
SCALES = ["S100", "ME"]


class Analysis:
    @classmethod
    def magnitude_estimates(cls, data: pd.DataFrame, num_scenarios: int = 5) -> pd.DataFrame:
        """Retrieves the magnitude estimates from the data.

        Args:
            data (pd.DataFrame): original dataframe object containing
            all survey data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: new dataframe object that contains
            all question codes with original (unnormalized) magnitude
            estimate values.
        """
        df = pd.DataFrame()

        for _index, row in data.iterrows():
            r = {}
            for type in TYPES:
                for i in range(1, num_scenarios + 1):
                    decision = cls.__get_value(row, "ME", type, i, "s")
                    value = cls.__get_value(row, "ME", type, i, "v")
                    me = cls.__convert_me(decision, value)
                    r[f"ME{type}{i}"] = me

            df = df.append(r, ignore_index=True)

        return df

    @classmethod
    def s100_values(cls, data: pd.DataFrame, num_scenarios: int = 5) -> pd.DataFrame:
        """Retrieves the 100-level scale values from the data.

        Args:
            data (pd.DataFrame): original dataframe object containing
            all survey data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: new dataframe object that contains
            all question codes with original 100-level scale values.
        """
        df = pd.DataFrame()

        for _index, row in data.iterrows():
            r = {}
            for type in TYPES:
                for i in range(1, num_scenarios + 1):
                    decision = cls.__get_value(row, "S100", type, i, "s")
                    agree_value = cls.__get_value(
                        row, "S100", type, i, "a\[SQ001\]")
                    disagree_value = cls.__get_value(
                        row, "S100", type, i, "d\[SQ001\]")
                    v100 = cls.__convert_100(
                        decision, agree_value, disagree_value)
                    r[f"S100{type}{i}"] = v100

            df = df.append(r, ignore_index=True)

        return df

    @classmethod
    def normalize(cls, data: pd.DataFrame, magnitude_estimates: pd.DataFrame) -> pd.DataFrame:
        """Converts the magnitude_estimates dataframe to a normalized one.

        Args:
            data (pd.DataFrame): the original data is needed to retrieve the calibration values.
            magnitude_estimates (pd.DataFrame): dataframe with unnormalized magnitude estimates.

        Returns:
            pd.DataFrame:  dataframe with normalized magnitude estimates.
        """
        new_df = pd.DataFrame()

        for index, row in data.iterrows():
            mes = magnitude_estimates.iloc[[index]]
            pivot = cls.pivot_value(row)
            normalized_mes = mes.div(pivot)
            new_df = new_df.append(normalized_mes, ignore_index=True)

        return new_df

    @classmethod
    def convert_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Convert the original data to a dataframe that consists
        of all normalized magnitude estimates and 100-level scale values.

        Args:
            data (pd.DataFrame): 

        Returns:
            pd.DataFrame: dataframe that consists of all normalized 
            magnitude estimates and 100-level scale values.
        """
        mes = cls.magnitude_estimates(data)
        normalized_mes = cls.normalize(data, mes)
        s100 = cls.s100_values(data)
        return pd.concat([normalized_mes, s100], axis=1)

    @classmethod
    def print_means(cls, data: pd.DataFrame) -> None:
        """Prints the mean cost values of each scenario type and scale.

        Args:
            data (pd.DataFrame): the converted data.
        """
        print("===================")
        for scale in SCALES:
            print(f"{scale} scale")
            for type in TYPES:
                print(type, cls.calculate_mean(data, "ME", type))
            print("===================")

    @classmethod
    def print_reliabilities(cls, data: pd.DataFrame) -> None:
        """Prints the reliability values of each scenario type and scale.
         The reliability values are Krippendorff's alpha values.

        Args:
            data (pd.DataFrame): the converted data.
        """
        print("===================")
        print("alpha complete data:", cls.reliability(data))
        print("===================")
        for scale in SCALES:
            print(f"{scale} scale")
            for type in TYPES:
                print(type, cls.reliability(data, scale=scale, type=type))
            print("===================")

    @staticmethod
    def reliability(data: pd.DataFrame, scale: str = '', type: str = '') -> float:
        """Calculates Krippendorffs's alpha values for the complete data 
        or for the filtered data if a scale and type filter is passed.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str, optional): 'ME' or 'S100'. Defaults to ''.
            type (str, optional): 'TP', 'TN', 'FP', 'FN', or 'REJ'. Defaults to ''.

        Returns:
            float: Krippendorff's alpha value.
        """

        if scale != '' or type != '':
            data = data.filter(regex=f"{scale}{type}", axis=1)

        data = data.values.tolist()

        try:
            alpha = krippendorff.alpha(reliability_data=data)
        except:
            # The value domain should consist of at least one item
            # If not, then catch exception and retun np.nan
            alpha = np.nan

        return alpha

    @staticmethod
    def plot_validity(data: pd.DataFrame) -> None:
        """Plots a correlation plot between 100-level scores 
        and the magnitude estimates.

        Args:
            data (pd.DataFrame): the converted data.
        """
        mes = data.filter(regex="ME", axis=1)
        s100 = data.filter(regex="S100", axis=1)
        mes = mes.mean().tolist()
        s100 = s100.mean().tolist()

        sns.regplot(x=mes, y=s100)
        plt.title("Correlation")
        plt.xlabel("ME")
        plt.ylabel("100-level")
        plt.show()

    @staticmethod
    def print_correlation(data: pd.DataFrame):
        """Prints the correlation between the 100-level scores
        and the magnitude estimates.

        Args:
            data (pd.DataFrame): the converted data.
        """
        mes = data.filter(regex="ME", axis=1)
        s100 = data.filter(regex="S100", axis=1)
        mes = mes.mean().tolist()
        s100 = s100.mean().tolist()

        print("Pearson", stats.pearsonr(mes, s100))
        print("Spearman", stats.spearmanr(mes, s100))
        print("Kendall", stats.kendalltau(mes, s100))

    @staticmethod
    def pivot_value(row: pd.Series) -> float:
        """Calculates the pivot value.

        Args:
            row (pd.Series): row from the original data.
            Each row represents one subject.

        Returns:
            float: the pivot value.
        """
        NAME = "G20Q51"
        str_dis = row.filter(regex=f"^{NAME}\[SQ001\]\.").values[0]
        dis = row.filter(regex=f"^{NAME}\[SQ002\]\.").values[0]
        som_dis = row.filter(regex=f"^{NAME}\[SQ003\]\.").values[0]
        som_agr = row.filter(regex=f"^{NAME}\[SQ005\]\.").values[0]
        agr = row.filter(regex=f"^{NAME}\[SQ006\]\.").values[0]
        str_agr = row.filter(regex=f"^{NAME}\[SQ007\]\.").values[0]
        calibration_vals = [str_dis, som_dis, dis, som_agr, agr, str_agr]
        absolute_cal_vals = [abs(val) for val in calibration_vals]
        return np.mean(absolute_cal_vals)

    @staticmethod
    def calculate_mean(data: pd.DataFrame, scale: str, type: str) -> float:
        """Calculates the mean cost for a specific
        scale and scenario type.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str): 'ME' or 'S100'.
            type (str): 'TP', 'TN', 'FP', 'FN', or 'REJ'.

        Returns:
            float: the mean cost value.
        """
        type_values = data.filter(regex=f"{scale}{type}", axis=1)
        column_means = type_values.mean()
        return column_means.mean()

    @staticmethod
    def __get_value(row, scale, type, index, question):
        return row.filter(regex=f"^{scale}{type}{index}{question}\.").values[0]

    @staticmethod
    def __convert_100(decision, agree_value, disagree_value):
        if decision == 'Agree':
            return agree_value
        elif decision == 'Disagree':
            return -disagree_value
        elif decision == 'Neutral':
            return 0

    @staticmethod
    def __convert_me(decision, value):
        if decision == 'Agree':
            return value
        elif decision == 'Disagree':
            return -value
        elif decision == 'Neutral':
            return 0

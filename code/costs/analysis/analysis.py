import numpy as np
import pandas as pd
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import matplotlib.ticker as mtick

TYPES = ["TP", "TN", "FP", "FN", "REJ"]
SCALES = ["ME", "S100"]


class Analysis:
    @classmethod
    def magnitude_estimates(cls, data: pd.DataFrame, num_scenarios: int = 4) -> pd.DataFrame:
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
    def s100_values(cls, data: pd.DataFrame, num_scenarios: int = 4) -> pd.DataFrame:
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
    def hatefulness(cls, data: pd.DataFrame, num_scenarios: int = 4) -> pd.DataFrame:
        """Retrieves the binary values of the hatefulness questions from the data.
        Each scenario contains one question about whether the tweet was considered hateful
        or non-hateful by the subject.

        Args:
            data (pd.DataFrame): original dataframe object containing all survey data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: new dataframe object that contains
            all question codes with boolean values about the hatefulness.
        """
        df = pd.DataFrame()

        for _index, row in data.iterrows():
            r = {}
            for scale in SCALES:
                for type in TYPES:
                    for i in range(1, num_scenarios + 1):
                        hatefulness = cls.__get_value(row, scale, type, i, "h")
                        hateful = cls.__convert_hatefulness(hatefulness)
                        r[f"Hateful_{scale}{type}{i}"] = hateful

            df = df.append(r, ignore_index=True)

        return df

    @classmethod
    def attention_checks(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Returns a dataframe that contains a column attention_checks_passed
        that indicates whether the subject has passed the attention check or not.

        Args:
            data (pd.DataFrame): original dataframe object containing all survey data.

        Returns:
            pd.DataFrame: new dataframe object that contains one column that indicates
            whether the subject has passed the attention checks or not.
        """
        df = pd.DataFrame()

        for _index, row in data.iterrows():
            attention1 = row.filter(regex="^attention1\.").values[0]
            attention2 = row.filter(regex="^attention2\.").values[0]
            attention_checks_passed = attention1 == "Blue" and attention2 == "Orange"

            df = df.append({'attention_checks_passed': attention_checks_passed}, ignore_index=True)

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
        hatefulness = cls.hatefulness(data)
        attention_checks = cls.attention_checks(data)
        return pd.concat([normalized_mes, s100, hatefulness, attention_checks], axis=1)

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

    @classmethod
    def plot_boxplots(cls, data: pd.DataFrame) -> None:
        """Plots boxplots of all individual scenarios.

        Args:
            data (pd.DataFrame): the converted data.
        """
        plot_data = cls.convert_to_boxplot_data(data)

        sns.boxplot(x="Scenario", y="(Dis)agreement",
                    hue="Scale", data=plot_data)
        sns.despine(offset=10, trim=True)
        plt.title("Boxplots of all questions")
        plt.xlabel("Scenario")
        plt.ylabel("(Dis)Agreement")
        plt.show()

    
    @classmethod
    def plot_hatefulness(cls, data: pd.DataFrame) -> None:
        """Plots the percentage of (non)hateful rated scenarios.

        Args:
            data (pd.DataFrame): the converted data.
        """        
        plot_data = cls.convert_to_stackedbar_data(data)
        sns.set(style='whitegrid')
        ax = plot_data.plot(kind='bar', stacked=True, x="Scenario")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.xlabel('Scenario')
        plt.ylabel('Percentage')
        plt.title('Percentage of (non)hateful rated scenarios')
        plt.show()

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

        if scale != '' and type != '':
            data = data.filter(regex=f"^{scale}{type}.*$", axis=1)
        elif scale != '' and type == '':
            data = data.filter(regex=f"^{scale}(TP|TN|FP|FN|REJ).*$", axis=1)
        elif scale == '' and type == '':
            data = data.filter(regex="^(ME|S100)(TP|TN|FP|FN|REJ).*$", axis=1)
        elif scale == '' and type != '':
            data = data.filter(regex=f"^(ME|S100){type}.*$", axis=1)

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
        mes = data.filter(regex="^ME.*$", axis=1)
        s100 = data.filter(regex="^S100.*$", axis=1)
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
        mes = data.filter(regex="^ME.*$", axis=1)
        s100 = data.filter(regex="^S100.*$", axis=1)
        mes = mes.mean().tolist()
        s100 = s100.mean().tolist()
        cohens_d = (np.mean(mes) - np.mean(s100)) / (np.sqrt((np.std(mes) ** 2 + np.std(s100) ** 2) / 2))
        print("Cohen's d", cohens_d)
        print("Unpaired T-test", stats.ttest_ind(mes, s100))
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
    def append_durations(data: pd.DataFrame) -> pd.DataFrame:
        """Adds duration column to the dataframe and replaces all duration
        values with None when the subject is 3 times the standard deviation
        below the mean duration.

        Args:
            data (pd.DataFrame): the original input data.

        Returns:
            pd.DataFrame: the original input data with one additional duration column.
        """
        durations = []
        for index, row in data.iterrows():
            startdate = row.filter(regex="startdate\.").values[0]
            submitdate = row.filter(regex="submitdate\.").values[0]
            startdate = datetime.strptime(startdate, '%Y-%m-%d %H:%M:%S')
            submitdate = datetime.strptime(submitdate, '%Y-%m-%d %H:%M:%S')
            duration = submitdate - startdate
            durations.append(duration.total_seconds())

        mean = np.mean(durations)
        std = np.std(durations)
        min_value = mean - 3 * std
        data['duration'] = durations
        return data.mask(data['duration'] < min_value)

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
        type_values = data.filter(regex=f"^{scale}{type}.*$", axis=1)
        column_means = type_values.mean()
        return round(column_means.mean(), 6)

    @staticmethod
    def convert_to_boxplot_data(data: pd.DataFrame) -> pd.DataFrame:
        """Converts the converted data to a new dataframe that is suitable
        for plotting the boxplots of all individual scenarios.

        Args:
            data (pd.DataFrame): the converted data.

        Returns:
            pd.DataFrame: converted to boxplot suitable data with three columns:
            (dis)agreement, scale, and scenario.
        """        
        data = data.filter(regex="^(ME|S100).*$", axis=1)
        question_names = data.columns.values.tolist()
        plot_data = []
        for index, question in enumerate(question_names):
            values = data[question]

            if question.startswith("ME"):
                scale = "ME"
                type = question.replace("ME", "")
            elif question.startswith("S100"):
                scale = "100-level"
                type = question.replace("S100", "")
            for value in values:
                plot_data.append([value, scale, type])

        return pd.DataFrame(plot_data, columns=["(Dis)agreement", "Scale", "Scenario"])

    @staticmethod
    def convert_to_stackedbar_data(data: pd.DataFrame) -> pd.DataFrame:
        """Converts the converted data to a new dataframe that is suitable
        for plotting the stacked bars with the percentages of (non)hateful
        rated scenarios.

        Args:
            data (pd.DataFrame): the converted data.

        Returns:
            pd.DataFrame: converted to stacked bar suitable data with three columns:
            scenario, hateful (percentage), and not hateful (percentage).
        """        
        data = data.filter(regex="^Hateful_(ME|S100).*$", axis=1)
        question_names = data.columns.values.tolist()
        plot_data = []

        for index, question in enumerate(question_names):
            values = data[question].tolist()
            count_hateful = sum(1 for value in values if value == True)
            total = len(values)
            percentage_hateful = round((count_hateful / total) * 100.0, 2)
            percentage_non_hateful = 100.0 - percentage_hateful
            plot_data.append([question.replace("Hateful_", ""), percentage_hateful, percentage_non_hateful])

        return pd.DataFrame(plot_data, columns=["Scenario", "Hateful", "Not hateful"])


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

    @staticmethod
    def __convert_hatefulness(hatefulness):
        if hatefulness == 'Hateful':
            return True
        elif hatefulness == 'Not hateful':
            return False
  
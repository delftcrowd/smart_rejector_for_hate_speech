from typing import List
import numpy as np
import pandas as pd
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import matplotlib.ticker as mtick
from string import digits

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
                        row, "S100", type, i, r"a\[SQ001\]")
                    disagree_value = cls.__get_value(
                        row, "S100", type, i, r"d\[SQ001\]")
                    v100 = cls.__convert_100(
                        decision, agree_value, disagree_value)
                    r[f"S100{type}{i}"] = v100

            df = df.append(r, ignore_index=True)

        return df

    @classmethod
    def hatefulness(cls, data: pd.DataFrame, scales: List[str] = SCALES, num_scenarios: int = 4) -> pd.DataFrame:
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
            for scale in scales:
                for type in TYPES:
                    for i in range(1, num_scenarios + 1):
                        hatefulness = cls.__get_value(row, scale, type, i, "h")
                        hateful = cls.__convert_hatefulness(hatefulness)
                        r[f"Hateful_{type}{i}"] = hateful

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
            attention1 = row.filter(regex=r"^attention1\.").values[0]
            attention2 = row.filter(regex=r"^attention2\.").values[0]
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
            pivot = np.max(np.abs(mes.values))
            normalized_mes = mes.div(pivot)
            new_df = new_df.append(normalized_mes, ignore_index=True)

        return new_df

    @classmethod
    def convert_both_scales_data(cls, data: pd.DataFrame, num_scenarios: int = 4) -> pd.DataFrame:
        """Convert the original data to a dataframe that consists
        of all normalized magnitude estimates and 100-level scale values.

        Args:
            data (pd.DataFrame): the original data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: dataframe that consists of all normalized
            magnitude estimates and 100-level scale values.
        """
        mes = cls.magnitude_estimates(data=data, num_scenarios=num_scenarios)
        normalized_mes = cls.normalize(data, mes)
        s100 = cls.s100_values(data=data, num_scenarios=num_scenarios)
        hatefulness = cls.hatefulness(data)
        attention_checks = cls.attention_checks(data)
        return pd.concat([normalized_mes, s100, hatefulness, attention_checks], axis=1)

    @classmethod
    def convert_me_data(cls, data: pd.DataFrame, num_scenarios: int = 8) -> pd.DataFrame:
        """Convert the original data to a dataframe that consists
        of all normalized magnitude estimates.

        Args:
            data (pd.DataFrame): the original data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: dataframe that consists of all normalized
            magnitude estimates.
        """
        mes = cls.magnitude_estimates(data=data, num_scenarios=num_scenarios)
        normalized_mes = cls.normalize(data, mes)
        hatefulness = cls.hatefulness(data=data, scales=["ME"], num_scenarios=num_scenarios)
        attention_checks = cls.attention_checks(data)
        return pd.concat([normalized_mes, hatefulness, attention_checks], axis=1)

    @classmethod
    def convert_100_data(cls, data: pd.DataFrame, num_scenarios: int = 8) -> pd.DataFrame:
        """Convert the original data to a dataframe that consists
        of 100-level scale values.

        Args:
            data (pd.DataFrame): the original data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: dataframe that consists of all normalized
            100-level scale values.
        """
        s100 = cls.s100_values(data=data, num_scenarios=num_scenarios)
        hatefulness = cls.hatefulness(data=data, scales=["S100"], num_scenarios=num_scenarios)
        attention_checks = cls.attention_checks(data)
        return pd.concat([s100, hatefulness, attention_checks], axis=1)

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
                print(type, cls.calculate_mean(data=data, scale=scale, type=type))
            print("===================")

    @classmethod
    def print_reliabilities(cls, data: pd.DataFrame) -> None:
        """Prints the reliability values of each scenario type and scale.
         The reliability values are Krippendorff's alpha values.

        Args:
            data (pd.DataFrame): the converted data.
        """
        print("===================")
        for scale in SCALES:
            print(f"{scale} scale")
            print("Reliability scale: ", cls.reliability(data, scale=scale))
            for type in TYPES:
                print(type, cls.reliability(data, scale=scale, type=type))
            print("===================")

    @classmethod
    def plot_boxplots(cls, data: pd.DataFrame, scale: str = None, show_individual: bool = True) -> None:
        """Plots boxplots of all individual scenarios.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str, optional): 'ME' or 'S100'. If nothing is passed, then both are plotted. Defaults to None.
            show_individual (bool, optional): Whether to show one boxplot per scenario or not. Defaults to True.
        """
        if scale is None:
            plot_data = cls.convert_to_dual_boxplot_data(data=data, show_individual=show_individual)
            sns.boxplot(x="Scenario", y="(Dis)agreement", hue="Scale", data=plot_data)
            plt.title("Boxplots of all questions")
        else:
            plot_data = cls.convert_to_boxplot_data(data=data, show_individual=show_individual, scale=scale)
            sns.boxplot(x="Scenario", y="(Dis)agreement", data=plot_data)
            plt.title(f"Boxplots of all questions for the {scale} scale")

        sns.despine(offset=10, trim=True)
        plt.xlabel("Scenario")
        plt.ylabel("(Dis)Agreement")
        plt.xticks(rotation=90)
        plt.show()

    @classmethod
    def plot_hatefulness(cls, data: pd.DataFrame) -> None:
        """Plots the percentage of (non)hateful rated scenarios.

        Args:
            data (pd.DataFrame): the converted data.
        """
        plot_data = cls.convert_to_stackedbar_data(data)
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
            column = f"^{scale}{type}.*$"
        elif scale != '' and type == '':
            column = f"^{scale}(TP|TN|FP|FN|REJ).*$"

        if scale == "ME":
            level_of_measurement = "ratio"
        elif scale == "S100":
            level_of_measurement = "interval"

        data = data.filter(regex=column, axis=1).values.tolist()

        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement=level_of_measurement)

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
    def print_statistics(data: pd.DataFrame):
        """Prints all statistics between the 100-level scores
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
        print("Shapiro Wilk normality test MES: ", stats.shapiro(mes))
        print("Shapiro Wilk normality test S100: ", stats.shapiro(s100))
        print("Bartlett's test for equal variances:  ", stats.bartlett(mes, s100))
        print("Mann-Whitney U test: ", stats.mannwhitneyu(mes, s100))
        print("Unpaired T-test: ", stats.ttest_ind(mes, s100))
        print("Pearson: ", stats.pearsonr(mes, s100))
        print("Spearman: ", stats.spearmanr(mes, s100))
        print("Kendall: ", stats.kendalltau(mes, s100))

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
            startdate = row.filter(regex=r"startdate\.").values[0]
            submitdate = row.filter(regex=r"submitdate\.").values[0]
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
    def any_failed_attention_checks(data: pd.DataFrame) -> bool:
        failed = data['attention_checks_passed'].loc[data['attention_checks_passed'] == 0.0].values
        return failed.size > 0

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

    @classmethod
    def convert_to_dual_boxplot_data(cls, data: pd.DataFrame, show_individual: bool) -> pd.DataFrame:
        """Converts the converted data to a new dataframe that is suitable
        for plotting the boxplots of all individual scenarios.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str): 'ME' or 'S100'. If nothing is passed, then both are plotted. Defaults to None.
            show_individual (bool): Whether to show one boxplot per scenario or not. Defaults to True.

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
                multiplier = 100
            elif question.startswith("S100"):
                scale = "100-level"
                type = question.replace("S100", "")
                multiplier = 1

            # Important: the magnitude estimates are multiplied by 100 (since they are normalized)
            # so that the boxplots can more easily be compared between both scale types.
            for value in values:
                plot_data.append([value * multiplier, type, scale])

        if not show_individual:
            cls.__remove_question_numbers_from_plot_data(plot_data)

        return pd.DataFrame(plot_data, columns=["(Dis)agreement", "Scenario", "Scale"])

    @classmethod
    def convert_to_boxplot_data(cls, data: pd.DataFrame, scale: str, show_individual: bool) -> pd.DataFrame:
        """Converts the converted data to a new dataframe that is suitable
        for plotting the boxplots of all individual scenarios.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str): 'ME' or 'S100'. If nothing is passed, then both are plotted. Defaults to None.
            show_individual (bool): Whether to show one boxplot per scenario or not. Defaults to True.

        Returns:
            pd.DataFrame: converted to boxplot suitable data with three columns:
            (dis)agreement, and scenario.
        """
        data = data.filter(regex=fr"^{scale}.*$", axis=1)
        question_names = data.columns.values.tolist()
        plot_data = []
        for index, question in enumerate(question_names):
            values = data[question]
            type = question.replace(scale, "")

            for value in values:
                plot_data.append([value, type])

        if not show_individual:
            cls.__remove_question_numbers_from_plot_data(plot_data)

        return pd.DataFrame(plot_data, columns=["(Dis)agreement", "Scenario"])

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
        data = data.filter(regex="^Hateful_.*$", axis=1)
        question_names = data.columns.values.tolist()
        plot_data = []
        for index, question in enumerate(question_names):
            values = data[question].tolist()
            count_hateful = sum(1 for value in values if value == 1.0)
            total = len(values)
            percentage_hateful = round((count_hateful / total) * 100.0, 2)
            percentage_non_hateful = 100.0 - percentage_hateful
            plot_data.append([question.replace("Hateful_", ""), percentage_hateful, percentage_non_hateful])

        return pd.DataFrame(plot_data, columns=["Scenario", "Hateful", "Not hateful"])

    @staticmethod
    def __get_value(row, scale, type, index, question):
        return row.filter(regex=fr"^{scale}{type}{index}{question}\.").values[0]

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

    @staticmethod
    def __remove_question_numbers_from_plot_data(data: pd.DataFrame) -> pd.DataFrame:
        for index, row in enumerate(data):
            # Remove the question number, e.g. 'TP1' becomes 'TP'
            row[1] = row[1].translate(str.maketrans('', '', digits))

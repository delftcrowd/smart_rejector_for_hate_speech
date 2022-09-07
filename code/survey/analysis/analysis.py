from typing import List
import numpy as np
import pandas as pd
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy import stats
from datetime import datetime
import matplotlib.ticker as mtick
from string import digits
import math
import itertools

TYPES = ["TP", "TN", "FP", "FN", "REJ"]
SCALES = ["ME", "S100"]
PROLIFIC_COLUMN = "prolificid. "


class Analysis:
    @classmethod
    def magnitude_estimates(
        cls, data: pd.DataFrame, num_scenarios: int = 4
    ) -> pd.DataFrame:
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
                    r[f"{type}{i}"] = me

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
                    agree_value = cls.__get_value(row, "S100", type, i, r"a\[SQ001\]")
                    disagree_value = cls.__get_value(
                        row, "S100", type, i, r"d\[SQ001\]"
                    )
                    v100 = cls.__convert_100(decision, agree_value, disagree_value)
                    r[f"{type}{i}"] = v100

            df = df.append(r, ignore_index=True)

        return df

    @classmethod
    def hatefulness(
        cls, data: pd.DataFrame, scale: str, num_scenarios: int = 4
    ) -> pd.DataFrame:
        """Retrieves the binary values of the hatefulness questions from the data.
        Each scenario contains one question about whether the tweet was considered hateful
        or non-hateful by the subject.

        Args:
            data (pd.DataFrame): original dataframe object containing all survey data.
            scale (str): 'ME' or 'S100'.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: new dataframe object that contains
            all question codes with boolean values about the hatefulness.
        """
        df = pd.DataFrame()

        for _index, row in data.iterrows():
            r = {}
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

            df = df.append(
                {"attention_checks_passed": attention_checks_passed}, ignore_index=True
            )

        return df

    @classmethod
    def normalize(
        cls, magnitude_estimates: pd.DataFrame, apply_log: bool = False
    ) -> pd.DataFrame:
        """Converts the magnitude_estimates dataframe to a normalized one.

        Args:
            magnitude_estimates (pd.DataFrame): dataframe with unnormalized magnitude estimates.
            apply_log (bool, optional): Whether we need to apply a log value to the absolute values. Defaults to False.

        Returns:
            pd.DataFrame:  dataframe with normalized magnitude estimates.
        """
        new_df = pd.DataFrame()

        for index, row in magnitude_estimates.iterrows():
            if apply_log is True:
                log_row = row.apply(cls.abs_log)
                pivot = np.max(np.abs(log_row.values))
                normalized_mes = log_row.div(pivot)
            else:
                pivot = np.max(np.abs(row.values))
                normalized_mes = row.div(pivot)
            new_df = new_df.append(normalized_mes, ignore_index=True)

        return new_df

    @classmethod
    def convert_me_data(
        cls, data: pd.DataFrame, num_scenarios: int = 8
    ) -> pd.DataFrame:
        """Convert the original data to a dataframe that consists
        of all normalized magnitude estimates.

        Args:
            data (pd.DataFrame): the original data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: dataframe that consists of all normalized
            magnitude estimates.
        """
        prolific_ids = data.loc[:, PROLIFIC_COLUMN]
        mes = cls.magnitude_estimates(data=data, num_scenarios=num_scenarios)
        normalized_mes = cls.normalize(mes).mul(100)
        hatefulness = cls.hatefulness(
            data=data, scale="ME", num_scenarios=num_scenarios
        )
        attention_checks = cls.attention_checks(data)
        return pd.concat(
            [prolific_ids, normalized_mes, hatefulness, attention_checks], axis=1
        )

    @classmethod
    def convert_100_data(
        cls, data: pd.DataFrame, num_scenarios: int = 8
    ) -> pd.DataFrame:
        """Convert the original data to a dataframe that consists
        of 100-level scale values.

        Args:
            data (pd.DataFrame): the original data.
            num_scenarios (int, optional): The number of scenarios per type. Defaults to 5.

        Returns:
            pd.DataFrame: dataframe that consists of all normalized
            100-level scale values.
        """
        prolific_ids = data.loc[:, PROLIFIC_COLUMN]
        s100 = cls.s100_values(data=data, num_scenarios=num_scenarios)
        hatefulness = cls.hatefulness(
            data=data, scale="S100", num_scenarios=num_scenarios
        )
        attention_checks = cls.attention_checks(data)
        return pd.concat([prolific_ids, s100, hatefulness, attention_checks], axis=1)

    @classmethod
    def print_means(cls, data: pd.DataFrame) -> None:
        """Prints the mean cost values of each scenario type and scale.

        Args:
            data (pd.DataFrame): the converted data.
        """
        print("===================")
        for type in TYPES:
            print(type, cls.calculate_mean(data=data, type=type))
        print("===================")

    @classmethod
    def print_reliabilities(cls, data: pd.DataFrame, scale: str) -> None:
        """Prints the reliability values of each scenario type and scale.
         The reliability values are Krippendorff's alpha values.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str): 'ME' or 'S100'.
        """
        print("===================")
        print("Reliability scale: ", cls.reliability(data=data, scale=scale))
        for type in TYPES:
            print(type, cls.reliability(data, scale=scale, type=type))
        print("===================")

    @classmethod
    def plot_dual_boxplots(
        cls,
        data_mes: pd.DataFrame,
        data_s100: pd.DataFrame,
        show_individual: bool = True,
    ) -> None:
        """Plots boxplots of all individual scenarios.

        Args:
            data_mes (pd.DataFrame): the converted mes data.
            data_s100 (pd.DataFrame): the converted 100-level data.
            show_individual (bool, optional): Whether to show one boxplot per scenario or not. Defaults to True.
        """
        plot_data = cls.convert_to_dual_boxplot_data(
            data_mes=data_mes, data_s100=data_s100, show_individual=show_individual
        )
        sns.boxplot(x="Scenario", y="(Dis)agreement", hue="Scale", data=plot_data)
        # plt.title("Boxplots of all scenarios")
        sns.despine(offset=10, trim=True)
        plt.xlabel("Scenario")
        plt.ylabel("(Dis)Agreement")
        plt.xticks(rotation=90)
        plt.savefig("dual-boxplots.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    @classmethod
    def plot_boxplots(
        cls, data: pd.DataFrame, scale_title: str, show_individual: bool = True
    ) -> None:
        """Plots boxplots of all individual scenarios.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str, optional): 'ME' or 'S100'. If nothing is passed, then both are plotted. Defaults to None.
            show_individual (bool, optional): Whether to show one boxplot per scenario or not. Defaults to True.
        """
        if show_individual:
            indv_title = "INDV"
        else:
            indv_title = "GRP"

        plot_data = cls.convert_to_boxplot_data(
            data=data, show_individual=show_individual
        )
        sns.boxplot(x="Scenario", y="(Dis)agreement", data=plot_data, color="#8c8c8c")
        # plt.title(f"Boxplots of all scenarios for the {scale_title} scale")

        sns.despine(offset=10, trim=True)
        plt.xlabel("Scenario")
        plt.ylabel("(Dis)Agreement")
        plt.xticks(rotation=90)
        plt.savefig(
            f"boxplots-{scale_title}-{indv_title}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.show()

    @classmethod
    def plot_hatefulness(cls, data_mes: pd.DataFrame, data_s100: pd.DataFrame) -> None:
        """Plots the percentage of (non)hateful rated scenarios.

        Args:
            data_mes (pd.DataFrame): the converted mes data.
            data_s100 (pd.DataFrame): the converted 100-level data.
        """
        data = data_mes.append(data_s100)
        plot_data = cls.convert_to_stackedbar_data(data)
        ax = plot_data.plot(kind="bar", stacked=True, x="Scenario")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.xlabel("Scenario")
        plt.ylabel("Percentage")
        # plt.title("Percentage of (non)hateful rated scenarios")
        plt.savefig("hatefulness.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    @classmethod
    def print_question_statistics(
        cls, data1: pd.DataFrame, data2: pd.DataFrame
    ) -> pd.DataFrame:
        """Prints all statistics between two datasets for each question.

        Args:
            data1 (pd.DataFrame): the first dataset.
            data2 (pd.DataFrame): the second dataset.

        Returns:
            pd.DataFrame: dataframe containing the p values of the question statistics.
        """
        all_scores, question_names = cls.convert_to_question_scores(data1, data2)

        df = pd.DataFrame(columns=["Scenario", "Mann-Whitney U", "T-test"])

        for index, question in enumerate(question_names):
            print("=================================")
            print("Question: ", question)
            print("=================================")
            question_scores = all_scores[index]

            for index, s in enumerate(question_scores):
                shapiro = stats.shapiro(s)
                if shapiro.pvalue >= 0.05:
                    print(f"Dataset {index} is normally distributed: ", shapiro)

            bartlett = stats.bartlett(*question_scores)
            mannwhitneyu = stats.mannwhitneyu(*question_scores)
            ttest_ind = stats.ttest_ind(*question_scores)

            if bartlett.pvalue >= 0.05:
                print("Variances are equal: ", bartlett)

            if mannwhitneyu.pvalue < 0.05:
                print("Significant difference: ", mannwhitneyu)
            else:
                print("No significant difference: ", mannwhitneyu)

            if ttest_ind.pvalue < 0.05:
                print("Significant difference: ", ttest_ind)
            else:
                print("No significant difference: ", ttest_ind)

            df = df.append(
                {
                    "Scenario": question,
                    "Mann-Whitney U": mannwhitneyu.pvalue,
                    "T-test": ttest_ind.pvalue,
                },
                ignore_index=True,
            )

        return df

    @classmethod
    def print_question_statistics_multiple(cls, *datasets):
        """Prints all statistics between all passed sample dataset lists for each question."""
        all_scores, question_names = cls.convert_to_question_scores(*datasets)

        df = pd.DataFrame(columns=["Scenario", "Kruskal-Wallis", "One-way ANOVA"])

        for index, question in enumerate(question_names):
            print("=================================")
            print("Question: ", question)
            print("=================================")
            question_scores = all_scores[index]

            for index, s in enumerate(question_scores):
                shapiro = stats.shapiro(s)
                if shapiro.pvalue >= 0.05:
                    print(f"Dataset {index} is normally distributed: ", shapiro)

            bartlett = stats.bartlett(*question_scores)
            kruskal = stats.kruskal(*question_scores)
            f_oneway = stats.f_oneway(*question_scores)

            if bartlett.pvalue >= 0.05:
                print("Variances are equal: ", bartlett)

            if kruskal.pvalue < 0.05:
                print("Significant difference between groups: ", kruskal)
                matrix = cls.nonparametric_pair_tests(question_scores)
                pd.DataFrame(matrix).to_csv(f"{question}_nonparametric.csv")
            else:
                print("Groups are equal: ", kruskal)

            if f_oneway.pvalue < 0.05:
                print("Significant difference between groups: ", f_oneway)
                matrix = cls.parametric_pair_tests(question_scores)
                pd.DataFrame(matrix).to_csv(f"{question}_parametric.csv")
            else:
                print("Groups are equal: ", f_oneway)

            df = df.append(
                {
                    "Scenario": question,
                    "Kruskal-Wallis": kruskal.pvalue,
                    "One-way ANOVA": f_oneway.pvalue,
                },
                ignore_index=True,
            )

        return df

    @staticmethod
    def nonparametric_pair_tests(
        question_scores: List[List[float]],
    ) -> List[List[float]]:
        """Conducts non-parametric tests between all pairs of groups.

        Args:
            question_scores (List[List[float]]): list of lists containing
            the scores of all participants per demographic group.

        Returns:
            List[List[float]]: pairwise corrected p-values.
        """
        pvalues = []
        pairs = []
        matrix = np.empty([len(question_scores), len(question_scores)])

        for (index1, data1), (index2, data2) in itertools.product(
            enumerate(question_scores), repeat=2
        ):
            mannwhitneyu = stats.mannwhitneyu(data1, data2)
            pvalues.append(mannwhitneyu.pvalue)
            pairs.append((index1, index2))

        (_, pvalues_corrected, _, _) = multipletests(pvalues, method="fdr_bh")

        for index, (dataset1_index, dataset2_index) in enumerate(pairs):
            pvalue = pvalues_corrected[index]
            matrix[dataset1_index][dataset2_index] = pvalue
            if pvalue < 0.05:
                print(
                    f"Significant difference (Mann-Whitney U) between dataset {dataset1_index} and {dataset2_index}: ",
                    pvalue,
                )
            else:
                print(
                    f"Groups are equal (Mann-Whitney U) between dataset {dataset1_index} and {dataset2_index}: ",
                    pvalue,
                )

        return matrix

    @staticmethod
    def parametric_pair_tests(question_scores: List[List[float]]):
        """Conducts parametric tests between all pairs of groups.

        Args:
            question_scores (List[List[float]]): list of lists containing
            the scores of all participants per demographic group.

        Returns:
            List[List[float]]: pairwise corrected p-values.
        """
        pvalues = []
        pairs = []
        matrix = np.empty([len(question_scores), len(question_scores)])

        for (index1, data1), (index2, data2) in itertools.product(
            enumerate(question_scores), repeat=2
        ):
            ttest_ind = stats.ttest_ind(data1, data2)
            pvalues.append(ttest_ind.pvalue)
            pairs.append((index1, index2))

        (_, pvalues_corrected, _, _) = multipletests(pvalues, method="fdr_bh")

        for index, (dataset1_index, dataset2_index) in enumerate(pairs):
            pvalue = pvalues_corrected[index]
            matrix[dataset1_index][dataset2_index] = pvalue
            if pvalue < 0.05:
                print(
                    f"Statistical difference (independent t-test) between dataset {dataset1_index} and {dataset2_index}: ",
                    pvalue,
                )
            else:
                print(
                    f"Groups are equal (independent t-test) between dataset {dataset1_index} and {dataset2_index}: ",
                    pvalue,
                )

        return matrix

    @staticmethod
    def filter_data(
        demo_data: pd.DataFrame, data: pd.DataFrame, column_name: str, column_value: any
    ) -> pd.DataFrame:
        """Filters the survey data on a specific column name and column value.

        Args:
            demo_data (pd.DataFrame): the demographics data.
            data (pd.DataFrame): the survey data.
            column_name (str): the column to filter on.
            column_value (any): the column value to filter on.

        Returns:
            pd.DataFrame: the filtered survey data.
        """
        demo_data_column = demo_data.groupby([column_name])
        if column_value in demo_data_column.groups.keys():
            filtered_demo_data = demo_data_column.get_group(column_value)
            filtered_ids = filtered_demo_data.loc[:, "Participant id"].tolist()
            return data.loc[data[PROLIFIC_COLUMN].isin(filtered_ids)].reset_index(
                drop=True
            )
        else:
            return pd.DataFrame()

    @staticmethod
    def filter_demographics_data(
        demo_data: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Filters the demographic data.

        It filters out all rows with prolific ids that do not occur in the survey data.

        Args:
            demo_data (pd.DataFrame): the demographics data to filter.
            data (pd.DataFrame): the survey data.

        Returns:
            pd.DataFrame: the filtered demographics data.
        """
        prolific_ids = data.loc[:, PROLIFIC_COLUMN].tolist()
        return demo_data.loc[
            demo_data["Participant id"].isin(prolific_ids)
        ].reset_index(drop=True)

    @staticmethod
    def reliability(data: pd.DataFrame, scale: str, type: str = "") -> float:
        """Calculates Krippendorffs's alpha values for the complete scale data
        or for the filtered data if a scale and type filter is passed.

        Args:
            data (pd.DataFrame): the converted data.
            scale (str): 'ME' or 'S100'.
            type (str, optional): 'TP', 'TN', 'FP', 'FN', or 'REJ'. Defaults to ''.

        Returns:
            float: Krippendorff's alpha value.
        """

        if type != "":
            column = f"^{type}.*$"
        else:
            column = "^(TP|TN|FP|FN|REJ).*$"

        if scale == "ME":
            level_of_measurement = "ratio"
        elif scale == "S100":
            level_of_measurement = "interval"

        data = data.filter(regex=column, axis=1).values.tolist()
        if len(data) > 0:
            return krippendorff.alpha(
                reliability_data=data, level_of_measurement=level_of_measurement
            )
        else:
            return None

    @staticmethod
    def plot_validity(data_mes: pd.DataFrame, data_s100: pd.DataFrame) -> None:
        """Plots a correlation plot between 100-level scores
        and the magnitude estimates.

        Args:
            data_mes (pd.DataFrame): the converted mes data.
            data_s100 (pd.DataFrame): the converted 100-level data.
        """
        mes = data_mes.filter(regex="^(TP|TN|FP|FN|REJ).*$", axis=1)
        s100 = data_s100.filter(regex="^(TP|TN|FP|FN|REJ).*$", axis=1)
        mes = mes.median().tolist()
        s100 = s100.median().tolist()

        sns.regplot(x=mes, y=s100)
        plt.xlabel("Magnitude Estimation")
        plt.ylabel("100-level")
        plt.tight_layout()
        plt.savefig("correlation.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    @staticmethod
    def print_scale_statistics(data_mes: pd.DataFrame, data_s100: pd.DataFrame):
        """Prints all statistics between the 100-level scores
        and the magnitude estimates.

        Args:
            data_mes (pd.DataFrame): the converted mes data.
            data_s100 (pd.DataFrame): the converted 100-level data.
        """
        mes = data_mes.filter(regex="^(TP|TN|FP|FN|REJ).*$", axis=1)
        s100 = data_s100.filter(regex="^(TP|TN|FP|FN|REJ).*$", axis=1)
        mes = mes.median().tolist()
        s100 = s100.median().tolist()

        cohens_d = (np.mean(mes) - np.mean(s100)) / (
            np.sqrt((np.std(mes) ** 2 + np.std(s100) ** 2) / 2)
        )
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
            submitdate = row.filter(regex=r"datestamp\.").values[0]
            startdate = datetime.strptime(startdate, "%Y-%m-%d %H:%M:%S")
            submitdate = datetime.strptime(submitdate, "%Y-%m-%d %H:%M:%S")
            duration = submitdate - startdate
            durations.append(duration.total_seconds())

        mean = np.mean(durations)
        std = np.std(durations)
        min_value = mean - 3 * std
        data["duration"] = durations
        return data.mask(data["duration"] < min_value)

    @staticmethod
    def any_failed_attention_checks(data: pd.DataFrame) -> bool:
        failed = (
            data["attention_checks_passed"]
            .loc[data["attention_checks_passed"] == 0.0]
            .values
        )
        return failed.size > 0

    @staticmethod
    def calculate_mean(data: pd.DataFrame, type: str) -> float:
        """Calculates the mean cost for a specific
        scale and scenario type.

        Args:
            data (pd.DataFrame): the converted data.
            type (str): 'TP', 'TN', 'FP', 'FN', or 'REJ'.

        Returns:
            float: the mean cost value.
        """
        type_values = data.filter(regex=f"^{type}.*$", axis=1)
        # Calculate the median for the individual questions since the distribution
        # of the scores is skewed.
        column_medians = type_values.median()
        return column_medians.mean()

    @classmethod
    def convert_to_dual_boxplot_data(
        cls, data_mes: pd.DataFrame, data_s100: pd.DataFrame, show_individual: bool
    ) -> pd.DataFrame:
        """Converts the converted data to a new dataframe that is suitable
        for plotting the boxplots of all individual scenarios.

        Args:
            data_mes (pd.DataFrame): the converted mes data.
            data_s100 (pd.DataFrame): the converted 100-level data.
            show_individual (bool): Whether to show one boxplot per scenario or not. Defaults to True.

        Returns:
            pd.DataFrame: converted to boxplot suitable data with three columns:
            (dis)agreement, scale, and scenario.
        """
        data_types = data_mes.filter(regex="^(TP|TN|FP|FN|REJ).*$", axis=1)
        question_names = data_types.columns.values.tolist()
        plot_data = []
        for question in question_names:
            mes_values = data_mes[question]
            s100_values = data_s100[question]

            for value in mes_values:
                plot_data.append([value, question, "ME"])

            for value in s100_values:
                plot_data.append([value, question, "100-level"])

        if not show_individual:
            cls.__remove_question_numbers_from_plot_data(plot_data)

        return pd.DataFrame(plot_data, columns=["(Dis)agreement", "Scenario", "Scale"])

    @classmethod
    def convert_to_boxplot_data(
        cls, data: pd.DataFrame, show_individual: bool
    ) -> pd.DataFrame:
        """Converts the converted data to a new dataframe that is suitable
        for plotting the boxplots of all individual scenarios.

        Args:
            data (pd.DataFrame): the converted data.
            show_individual (bool): Whether to show one boxplot per scenario or not. Defaults to True.

        Returns:
            pd.DataFrame: converted to boxplot suitable data with three columns:
            (dis)agreement, and scenario.
        """
        data = data.filter(regex="^(TP|TN|FP|FN|REJ).*$", axis=1)
        question_names = data.columns.values.tolist()
        plot_data = []
        for index, question in enumerate(question_names):
            values = data[question]

            for value in values:
                plot_data.append([value, question])

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
            plot_data.append(
                [
                    question.replace("Hateful_", ""),
                    percentage_hateful,
                    percentage_non_hateful,
                ]
            )

        return pd.DataFrame(plot_data, columns=["Scenario", "Hateful", "Not hateful"])

    @staticmethod
    def convert_to_question_scores(*datasets):
        """Converts multiple datasets to a list containing the scores per question.

        It also returns the list of question names for convenience.
        """
        data = []
        for dataset in datasets:
            questions = dataset.filter(regex="^(TP|TN|FP|FN|REJ).*$", axis=1)
            data.append(questions)

        all_scores = []
        question_names = data[0].columns.values.tolist()
        for question in question_names:
            question_scores = list(map(lambda s: s[question].to_list(), data))
            all_scores.append(question_scores)

        return all_scores, question_names

    @staticmethod
    def group_scenario_scores(data: pd.DataFrame) -> pd.DataFrame:
        """Groups the scores to the individual scenarios of each participant
        to mean group scores (so mean values of TP, TN, FP, FN, and rejected predictions).

        Args:
            data (pd.DataFrame): the original data with scores to individual scenarios.

        Returns:
            pd.DataFrame: the converted data where the scores are aggregated per scenario type.
        """
        prolific_ids = data[PROLIFIC_COLUMN]
        grouped_data = {PROLIFIC_COLUMN: prolific_ids}
        for type in TYPES:
            type_data = data.filter(regex=f"^{type}.*$", axis=1)
            type_means = type_data.mean(axis=1).tolist()
            grouped_data[type] = type_means

        return pd.DataFrame(grouped_data)

    @staticmethod
    def __get_value(row, scale, type, index, question):
        return row.filter(regex=rf"^{scale}{type}{index}{question}\.").values[0]

    @staticmethod
    def __convert_100(decision, agree_value, disagree_value):
        if decision == "Agree":
            return agree_value
        elif decision == "Disagree":
            return -disagree_value
        elif decision == "Neutral":
            return 0

    @staticmethod
    def __convert_me(decision, value):
        if decision == "Agree":
            return value
        elif decision == "Disagree":
            return -value
        elif decision == "Neutral":
            return 0

    @staticmethod
    def __convert_hatefulness(hatefulness):
        if hatefulness == "Hateful":
            return True
        elif hatefulness == "Not hateful":
            return False

    @staticmethod
    def __remove_question_numbers_from_plot_data(data: pd.DataFrame) -> pd.DataFrame:
        for index, row in enumerate(data):
            # Remove the question number, e.g. 'TP1' becomes 'TP'
            row[1] = row[1].translate(str.maketrans("", "", digits))

    @staticmethod
    def abs_log(value):
        if value == 0:
            return 0
        elif value < 0:
            return -math.log10(abs(value))
        elif value > 0:
            return math.log10(value)

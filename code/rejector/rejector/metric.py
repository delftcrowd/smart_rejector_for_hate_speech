from __future__ import annotations
from rejector.values import Values
from rejector.prediction import Prediction
from typing import List, Dict, Tuple
import numpy as np
from matplotlib import pyplot
from rejector.pdfs import PDFs
import seaborn as sns


class Metric():
    """Provides helper functions for handling predictions and calculating sets of
    TP, TN, FP, and FN.
    """

    def __init__(self, values: Values, predictions: List[Prediction],
                 estimator_conf: Dict[str, Dict[str, object]] = None) -> None:
        """Initializes the metric

        Args:
            values (Values): The values of TP, TN, FP, and FN.
            predictions (List[Prediction]): The list of predictions.
            estimator_conf (Dict[str, Dict[str, object]], optional): Dictionary that contains
                the KDE params. Defaults to {} and finds the optimal
                params automatically. Defaults to None.
        """
        self.values = values
        self.predictions = predictions
        self.estimator_conf = estimator_conf
        self.pdfs = PDFs(self.predictions, self.estimator_conf)

    def calculate_effectiveness(self, threshold: float, use_pdf: bool = True) -> float:
        """Calculates the effectiveness of a model for a specific threshold and value values

        Args:
            threshold (float): The reliability threshold value.
            use_pdf (bool, optional): Whether to use the Probability Density Functions or not.
            Otherwise, we simply count the number of predictions above/below the threshold. Defaults to True.

        Returns:
            float: The effectiveness of the model.
        """
        value_TP = self.values.value_TP
        value_TN = self.values.value_TN
        value_FP = self.values.value_FP
        value_FN = self.values.value_FN
        value_rejection = self.values.value_rejection

        if use_pdf:
            tps = self.pdfs.tps
            tns = self.pdfs.tns
            fps = self.pdfs.fps
            fns = self.pdfs.fns

            # Keep more simplistic metric, just in case
            # return value_TP * tps.integral(min=threshold, max=1.0) \
            #     + value_TN * tns.integral(min=threshold, max=1.0) \
            #     - value_FP * fps.integral(min=threshold, max=1.0) \
            #     - value_FN * fns.integral(min=threshold, max=1.0) \
            #     - value_rejection * tps.integral(min=0, max=threshold) \
            #     - value_rejection * tns.integral(min=0, max=threshold) \
            #     - value_rejection * fps.integral(min=0, max=threshold) \
            #     - value_rejection * fns.integral(min=0, max=threshold)

            return (value_TP + value_rejection) * tps.integral(min=threshold, max=1.0) \
                + (value_TN + value_rejection) * tns.integral(min=threshold, max=1.0) \
                + (value_rejection - value_FP) * fps.integral(min=threshold, max=1.0) \
                + (value_rejection - value_FN) * fns.integral(min=threshold, max=1.0) \
                - (value_rejection + value_TP) * tps.integral(min=0, max=threshold) \
                - (value_rejection + value_TN) * tns.integral(min=0, max=threshold) \
                + (value_FP - value_rejection) * fps.integral(min=0, max=threshold) \
                + (value_FN - value_rejection) * fns.integral(min=0, max=threshold)

            # return value_TP * tps.integral(min=threshold, max=1.0) \
            #     + value_TN * tns.integral(min=threshold, max=1.0) \
            #     - value_FP * fps.integral(min=threshold, max=1.0) \
            #     - value_FN * fns.integral(min=threshold, max=1.0) \
            #     - value_TP * tps.integral(min=0, max=threshold) \
            #     - value_TN * tns.integral(min=0, max=threshold) \
            #     + value_FP * fps.integral(min=0, max=threshold) \
            #     + value_FN * fns.integral(min=0, max=threshold)
        else:
            tps = Prediction.set_of_tps(self.predictions)
            tns = Prediction.set_of_tns(self.predictions)
            fps = Prediction.set_of_fps(self.predictions)
            fns = Prediction.set_of_fns(self.predictions)

            # Keep more simplistic metric, just in case
            # return value_TP * Prediction.count_above_threshold(tps, threshold) \
            #     + value_TN * Prediction.count_above_threshold(tns, threshold) \
            #     - value_FP * Prediction.count_above_threshold(fps, threshold) \
            #     - value_FN * Prediction.count_above_threshold(fns, threshold) \
            #     - value_rejection * Prediction.count_below_threshold(tps, threshold) \
            #     - value_rejection * Prediction.count_below_threshold(tns, threshold) \
            #     - value_rejection * Prediction.count_below_threshold(fps, threshold) \
            #     - value_rejection * Prediction.count_below_threshold(fns, threshold)

            # Another alternative, only keep correct and incorrect into account.
            # return value_rejection * Prediction.count_above_threshold(tps, threshold) \
            #     + value_rejection * Prediction.count_above_threshold(tns, threshold) \
            #     - value_FP * Prediction.count_above_threshold(fps, threshold) \
            #     - value_FN * Prediction.count_above_threshold(fns, threshold) \
            #     - value_rejection * Prediction.count_below_threshold(tps, threshold) \
            #     - value_rejection * Prediction.count_below_threshold(tns, threshold) \
            #     + value_FP * Prediction.count_below_threshold(fps, threshold) \
            #     + value_FN * Prediction.count_below_threshold(fns, threshold)

            return (value_TP + value_rejection) * Prediction.count_above_threshold(tps, threshold) \
                + (value_TN + value_rejection) * Prediction.count_above_threshold(tns, threshold) \
                + (value_rejection - value_FP) * Prediction.count_above_threshold(fps, threshold) \
                + (value_rejection - value_FN) * Prediction.count_above_threshold(fns, threshold) \
                - (value_rejection + value_TP) * Prediction.count_below_threshold(tps, threshold) \
                - (value_rejection + value_TN) * Prediction.count_below_threshold(tns, threshold) \
                + (value_FP - value_rejection) * Prediction.count_below_threshold(fps, threshold) \
                + (value_FN - value_rejection) * Prediction.count_below_threshold(fns, threshold)

    def plot_pdfs(self) -> None:
        """Plots the Probability Density Functions for TP, TN, FP, and FN      
        """
        fig, axs = pyplot.subplots(2, 2)
        plot_conf = [{'plt_y': 0, 'plt_x': 0, 'data': self.pdfs.tps, 'title': "True Positives"},
                     {'plt_y': 0, 'plt_x': 1, 'data': self.pdfs.tns,
                         'title': "True Negatives"},
                     {'plt_y': 1, 'plt_x': 0, 'data': self.pdfs.fps,
                         'title': "False Positives"},
                     {'plt_y': 1, 'plt_x': 1, 'data': self.pdfs.fns, 'title': "False Negatives"}]

        for conf in plot_conf:
            reliability_values = list(
                map(lambda p: p.predicted_value, conf['data'].predictions))
            x_values = conf['data'].pdf_x
            y_values = conf['data'].pdf_y

            axs[conf['plt_y'], conf['plt_x']].hist(
                reliability_values, bins=50, density=True)
            axs[conf['plt_y'], conf['plt_x']].plot(x_values[:], y_values)
            axs[conf['plt_y'], conf['plt_x']].set_title(conf['title'])
            axs[conf['plt_y'], conf['plt_x']].set_xlabel(
                "Reliability value")
            axs[conf['plt_y'], conf['plt_x']].set_ylabel("Probability Density")

        pyplot.suptitle("Probability Density Functions for the sets of TP, TN, FP, and FN\n" +
                        "The orange line is the estimated PDF that is derived using Kernel Density Estimation by fitting " +
                        "it with the original data. The blue histogram is the probability density of the original data")
        pyplot.show()

    def plot_effectiveness(self, use_pdf: bool = False) -> None:
        """Plots the model's effectiveness.

        Args:
            use_pdf (bool, optional): Whether to use the Probability Density Functions or not.
            Otherwise, we simply count the number of predictions above/below the threshold. Defaults to True.
        """
        thresholds = np.linspace(0, 1, 1000)

        effectiveness_values = list(
            map(lambda t:  self.calculate_effectiveness(t, use_pdf=use_pdf), thresholds))

        (index, max_effectiveness) = self.maximum_effectiveness(effectiveness_values)

        pyplot.plot(thresholds, effectiveness_values)
        pyplot.plot(thresholds[index], max_effectiveness,
                    marker='o', markersize=3, color="red")
        pyplot.annotate(
            f'Maximum effectiveness: (Threshold: {round(thresholds[index], 4)}, Effectiveness: {round(max_effectiveness, 4)})',
            (thresholds[index],
             max_effectiveness))
        pyplot.xlabel("Rejection threshold (σ)")
        pyplot.ylabel("Effectiveness of the model (P(σ))")
        pyplot.title(
            "Measuring the model's effectiveness for different rejection thresholds\n" +
            f"value TP: {self.values.value_TP}, value TN: {self.values.value_TN}, value FP: {self.values.value_FP}, " +
            f"value FN: {self.values.value_FN}, value rejection: {self.values.value_rejection}")
        pyplot.show()

    @classmethod
    def plot_multiple_effectiveness(
            cls, metrics: List[Tuple[str, Metric]],
            filename: str,
            show_yaxis_title: bool,
            use_pdf: bool = False):
        thresholds = np.linspace(0, 1, 1000)
        colors = sns.color_palette("colorblind")
        for index, (label, metric) in enumerate(metrics):
            eff = list(map(lambda t: metric.calculate_effectiveness(t, use_pdf=use_pdf), thresholds))
            (max_index, max_eff) = cls.maximum_effectiveness(eff)
            pyplot.plot(thresholds[max_index], max_eff, color="black", zorder=2, marker="d",
                        markerfacecolor='None', markeredgecolor="black", linestyle='None', label="Optimal τ")
            pyplot.plot(thresholds, eff, color=colors[index], label=f"{label}", zorder=1)

        if show_yaxis_title:
            pyplot.ylabel("Total value of the model (V(τ))")
        
        pyplot.xlabel("Rejection threshold (τ)")
        pyplot.xlim([0.5, 1.0])
        handles, labels = pyplot.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        print(by_label.values())
        print(by_label.keys())
        pyplot.tight_layout()
        pyplot.legend(by_label.values(), by_label.keys())
        pyplot.savefig(filename, format='pdf', bbox_inches='tight')
        pyplot.show()

    @staticmethod
    def maximum_effectiveness(effectiveness_values: List[float]) -> float:
        """Returns the maximum effectiveness value along with its index

        Args:
            effectiveness_values (List[float]): The model's effectiveness values for different threshold values.

        Returns:
            float: The effectiveness of the model.
        """
        index = np.argmax(effectiveness_values)
        return index, effectiveness_values[index]

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

            return (value_TP + value_rejection) * tps.integral(min=threshold, max=1.0) \
                + (value_TN + value_rejection) * tns.integral(min=threshold, max=1.0) \
                + (value_rejection - value_FP) * fps.integral(min=threshold, max=1.0) \
                + (value_rejection - value_FN) * fns.integral(min=threshold, max=1.0) \
                - (value_rejection + value_TP) * tps.integral(min=0, max=threshold) \
                - (value_rejection + value_TN) * tns.integral(min=0, max=threshold) \
                + (value_FP - value_rejection) * fps.integral(min=0, max=threshold) \
                + (value_FN - value_rejection) * fns.integral(min=0, max=threshold)

        else:
            tps = Prediction.set_of_tps(self.predictions)
            tns = Prediction.set_of_tns(self.predictions)
            fps = Prediction.set_of_fps(self.predictions)
            fns = Prediction.set_of_fns(self.predictions)

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

    def print_optimal_threshold_stats(self) -> None:
        """Prints the statistics for the optimal rejection threshold.
        """
        thresholds = np.arange(0.5, 1.0, 0.001)

        effectiveness_values = list(
            map(lambda t: self.calculate_effectiveness(t, use_pdf=False), thresholds))

        (index, max_effectiveness) = self.maximum_effectiveness(effectiveness_values)
        correct_original = list(filter(lambda p: p.predicted_class == p.actual_class, self.predictions))
        optimal_threshold = thresholds[index]
        accepted = list(filter(lambda p: p.predicted_value >= optimal_threshold, self.predictions))
        correct_accepted = list(filter(lambda p: p.predicted_class == p.actual_class, accepted))
        rejected = list(filter(lambda p: p.predicted_value < optimal_threshold, self.predictions))
        print("Optimal threshold: ", optimal_threshold)
        print("Optimal V(threshold): ", max_effectiveness)
        print("Total value: ", max_effectiveness)
        print("Num accepted: ", len(accepted))
        print("Accuracy original model: ", len(correct_original) / len(self.predictions))
        print("Accuracy accepted: ", len(correct_accepted) / len(accepted))
        print("Num rejected: ", len(rejected))
        RR = len(rejected) / len(self.predictions)
        print("Percentage rejected: ", RR)

    @classmethod
    def plot_multiple_effectiveness(
            cls, metrics: List[Tuple[str, Metric]],
            filename: str,
            show_yaxis_title: bool,
            legend_loc: str,
            use_pdf: bool = False):
        """Plots the effectiveness (total value) of multiple metrics.

        Args:
            metrics (List[Tuple[str, Metric]]): List of tuples where the first element indicates
            the name of the model and the second element the metric itself.
            filename (str): The exportation filename.
            show_yaxis_title (bool): Whether to show the y-axis title or not.
            legend_loc (str): The location of the legend.
            use_pdf (bool, optional): Whether to use the PDFs or not. Defaults to False.
        """
        # We only plot the last 500 values (tau=0.5) since the confidence values are always greater than 0.5
        thresholds = np.arange(0.4, 1.0, 0.001)
        colors = sns.color_palette("colorblind")

        for index, (label, metric) in enumerate(metrics):
            eff = list(map(lambda t: metric.calculate_effectiveness(t, use_pdf=use_pdf), thresholds))
            (max_index, max_eff) = cls.maximum_effectiveness(eff[100:])
            pyplot.plot(
                thresholds[max_index + 100],
                max_eff, color="black", zorder=2, marker="d", markerfacecolor='None', markeredgewidth=3, markersize=10,
                markeredgecolor="black", linestyle='None', label="Optimal τ", linewidth=3)
            pyplot.plot(thresholds, eff, color=colors[index], label=f"{label}", zorder=1, linewidth=3)

        if show_yaxis_title:
            pyplot.ylabel("Total value of the model (V(τ))")

        pyplot.xlabel("Rejection threshold (τ)")
        pyplot.xlim([0.49, 1.01])
        handles, labels = pyplot.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        pyplot.tight_layout()
        pyplot.legend(by_label.values(), by_label.keys(), loc=legend_loc)
        pyplot.savefig(filename, format='pdf', bbox_inches='tight')
        pyplot.show()

    @classmethod
    def plot_multiple_confidence_densities(
            cls, metrics: List[Tuple[str, Metric]],
            filename: str,
            show_yaxis_title: bool,
            bw_adjust: float):
        colors = sns.color_palette("colorblind")
        fig, ax = pyplot.subplots(2, 2)

        for index, (label, metric) in enumerate(metrics):
            tps = Prediction.set_of_tps(metric.predictions)
            tns = Prediction.set_of_tns(metric.predictions)
            fps = Prediction.set_of_fps(metric.predictions)
            fns = Prediction.set_of_fns(metric.predictions)
            tps_conf = list(map(lambda p: p.predicted_value, tps))
            tns_conf = list(map(lambda p: p.predicted_value, tns))
            fps_conf = list(map(lambda p: p.predicted_value, fps))
            fns_conf = list(map(lambda p: p.predicted_value, fns))
            sns.kdeplot(tps_conf, color=colors[index], label=f"{label}",
                        zorder=1, linewidth=3, ax=ax[0, 0], bw_adjust=bw_adjust)
            sns.kdeplot(tns_conf, color=colors[index], label=f"{label}",
                        zorder=1, linewidth=3, ax=ax[0, 1], bw_adjust=bw_adjust)
            sns.kdeplot(fps_conf, color=colors[index], label=f"{label}",
                        zorder=1, linewidth=3, ax=ax[1, 0], bw_adjust=bw_adjust)
            sns.kdeplot(fns_conf, color=colors[index], label=f"{label}",
                        zorder=1, linewidth=3, ax=ax[1, 1], bw_adjust=bw_adjust)

        if show_yaxis_title:
            pyplot.ylabel("Density")

        pyplot.xlabel("Confidence value")
        pyplot.tight_layout()
        pyplot.legend()
        pyplot.savefig(filename, format='pdf', bbox_inches='tight')
        pyplot.show()

    @classmethod
    def plot_accuracy_rejection_curves(
            cls, metrics: List[Tuple[str, Metric]],
            filename: str,
            show_yaxis_title: bool,
            legend_loc: str,
            use_pdf: bool = False):
        X = np.arange(0.0, 1.0, 0.001)
        colors = sns.color_palette("colorblind")

        for index, (label, metric) in enumerate(metrics):
            sorted_predictions = sorted(metric.predictions, key=lambda x: x.predicted_value)
            y = []
            effectiveness = list(map(lambda t: metric.calculate_effectiveness(t, use_pdf=use_pdf), X))
            (max_index, max_eff) = cls.maximum_effectiveness(effectiveness)
            optimal_threshold = X[max_index]
            rejected = list(filter(lambda p: p.predicted_value < optimal_threshold, metric.predictions))
            optimal_t_rr = round(len(rejected) / len(metric.predictions), 2)

            for x in X:
                n = int(x * 100)
                accepted = sorted_predictions[n:]
                correct_accepted = list(filter(lambda p: p.predicted_class == p.actual_class, accepted))
                accuracy = len(correct_accepted) / len(accepted)
                y.append(accuracy)

            n = int(optimal_t_rr * 100)
            accepted = sorted_predictions[n:]
            correct_accepted = list(filter(lambda p: p.predicted_class == p.actual_class, accepted))
            optimal_t_accuracy = len(correct_accepted) / len(accepted)

            pyplot.plot(
                optimal_t_rr,
                optimal_t_accuracy, color="black", zorder=2, marker="d", markerfacecolor='None', markeredgewidth=3, markersize=10,
                markeredgecolor="black", linestyle='None', label="Optimal τ", linewidth=3)
            pyplot.plot(X, y, color=colors[index], label=f"{label}", zorder=1, linewidth=3)

        if show_yaxis_title:
            pyplot.ylabel("Accuracy")

        pyplot.xlabel("Rejection Rate")
        handles, labels = pyplot.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        pyplot.tight_layout()
        pyplot.legend(by_label.values(), by_label.keys(), loc=legend_loc)
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

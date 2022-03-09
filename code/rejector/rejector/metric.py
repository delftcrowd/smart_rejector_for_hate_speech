from rejector.costs import Costs
from rejector.prediction import Prediction
from typing import List, Dict
import numpy as np
from matplotlib import pyplot
from rejector.pdfs import PDFs


class Metric():
    """Provides helper functions for handling predictions and calculating sets of
    TP, TN, FP, and FN.
    """

    def __init__(self, costs: Costs, predictions: List[Prediction], estimator_conf: Dict[str, Dict[str, object]] = None) -> None:
        """Initializes the metric

        Args:
            costs (Costs): The costs of TP, TN, FP, and FN.
            predictions (List[Prediction]): The list of predictions.
            estimator_conf (Dict[str, Dict[str, object]], optional): Dictionary that contains
                the KDE params. Defaults to {} and finds the optimal
                params automatically. Defaults to None.
        """
        self.costs = costs
        self.predictions = predictions
        self.estimator_conf = estimator_conf
        self.pdfs = PDFs(self.predictions, self.estimator_conf)

    def calculate_effectiveness(self, threshold: float) -> float:
        """Calculates the effectiveness of a model for a specific threshold and cost values

        Args:
            threshold (float): The reliability threshold value.

        Returns:
            float: The effectiveness of the model.
        """
        return (self.costs.cost_FP - self.costs.cost_rejection) * \
            self.pdfs.fps.integral(threshold) \
            + (self.costs.cost_FN - self.costs.cost_rejection) * \
            self.pdfs.fns.integral(threshold) \
            - (self.costs.cost_TP + self.costs.cost_rejection) * \
            self.pdfs.tps.integral(threshold) \
            - (self.costs.cost_TN + self.costs.cost_rejection) * \
            self.pdfs.tns.integral(threshold)

    def caculate_derivative(self,  threshold: float) -> float:
        """Calculates the derivative of the effectivness for a specific threshold and
        costs

        Args:
            threshold (float): The reliability threshold value.

        Returns:
            float: The derivative of the effectiveness of the model.
        """
        return (self.costs.cost_FP - self.costs.cost_rejection) \
            * self.pdfs.fps.D(threshold) \
            + (self.costs.cost_FN - self.costs.cost_rejection) \
            * self.pdfs.fns.D(threshold) \
            - (self.costs.cost_TP + self.costs.cost_rejection) \
            * self.pdfs.tps.D(threshold) \
            - (self.costs.cost_TN + self.costs.cost_rejection) * \
            self.pdfs.tns.D(threshold)

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

    def plot_effectiveness(self) -> None:
        """Plots the model's effectiveness.
        """
        thresholds = np.linspace(0, 1, 1000)

        effectiveness_values = list(
            map(lambda t:  self.calculate_effectiveness(t), thresholds))

        (index, max_effectiveness) = self.maximum_effectiveness(effectiveness_values)

        pyplot.plot(thresholds, effectiveness_values)
        pyplot.plot(thresholds[index], max_effectiveness,
                    marker='o', markersize=3, color="red")
        pyplot.annotate(f'Maximum effectiveness: (Threshold: {round(thresholds[index], 4)}, Effectiveness: {round(max_effectiveness, 4)})', (
            thresholds[index], max_effectiveness))
        pyplot.xlabel("Rejection threshold (σ)")
        pyplot.ylabel("Effectiveness of the model (P(σ))")
        pyplot.title(
            "Measuring the model's effectiveness for different rejection thresholds\n" +
            f"cost TP: {self.costs.cost_TP}, cost TN: {self.costs.cost_TN}, cost FP: {self.costs.cost_FP}, " +
            f"cost FN: {self.costs.cost_FN}, cost rejection: {self.costs.cost_rejection}")
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

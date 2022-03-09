from original_rejector.costs import Costs
from original_rejector.prediction import Prediction
from typing import List, Dict
import numpy as np
from matplotlib import pyplot
from original_rejector.pdfs import PDFs


class Metric():
    """Provides helper functions for handling predictions.
    """

    def __init__(self, costs: Costs, predictions: List[Prediction], estimator_conf: Dict[str, Dict[str, object]] = None) -> None:
        """Initializes the metric

        Args:
            costs (Costs): The costs of correct/incorrect predictions and rejections.
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
        return (self.costs.cost_incorrect - self.costs.cost_rejection) * \
            self.pdfs.incorrect.integral(threshold) \
            - (self.costs.cost_correct + self.costs.cost_rejection) * \
            self.pdfs.correct.integral(threshold)

    def caculate_derivative(self, threshold: float) -> float:
        """Calculates the derivative of the effectivness for a specific threshold and
        costs

        Args:
            threshold (float): The reliability threshold value.

        Returns:
            float: The derivative of the effectiveness of the model.
        """
        return ((self.costs.cost_incorrect - self.costs.cost_rejection)
                / (self.costs.cost_rejection + self.costs.cost_correct)) \
            * self.pdfs.incorrect.D(threshold) - self.pdfs.correct.D(threshold)

    def plot_pdfs(self) -> None:
        """Plots the Probability Density Functions for TP, TN, FP, and FN      
        """
        fig, axs = pyplot.subplots(1, 2)
        plot_conf = [{'index': 0, 'data': self.pdfs.correct, 'title': "Correct"},
                     {'index': 1, 'data': self.pdfs.incorrect, 'title': "Incorrect"}]

        for conf in plot_conf:
            reliability_values = list(
                map(lambda p: p.predicted_value, conf['data'].predictions))
            x_values = conf['data'].pdf_x
            y_values = conf['data'].pdf_y

            axs[conf['index']].hist(
                reliability_values, bins=50, density=True)
            axs[conf['index']].plot(x_values[:], y_values)
            axs[conf['index']].set_title(conf['title'])
            axs[conf['index']].set_xlabel(
                "Reliability value")
            axs[conf['index']].set_ylabel("Probability Density")

        pyplot.suptitle("Probability Density Functions for the sets of correct and incorrect predictions\n" +
                        "The orange line is the estimated PDF that is derived using Kernel Density Estimation by fitting " +
                        "it with the original data. The blue histogram is the probability density of the original data")
        pyplot.show()

    def plot_effectiveness(self) -> None:
        """Plots the model's effectiveness.
        """
        thresholds = np.linspace(0, 1, 1000)

        effectiveness_values = list(
            map(lambda t:  self.calculate_effectiveness(t), thresholds))

        (index, max_effectiveness) = self.maximum_effectiveness(
            effectiveness_values)

        pyplot.plot(thresholds, effectiveness_values)
        pyplot.plot(thresholds[index], max_effectiveness,
                    marker='o', markersize=3, color="red")
        pyplot.annotate(f'Maximum effectiveness: (Threshold: {round(thresholds[index], 4)}, Effectiveness: {round(max_effectiveness, 4)})', (
            thresholds[index], max_effectiveness))
        pyplot.xlabel("Rejection threshold (σ)")
        pyplot.ylabel("Effectiveness of the model (P(σ))")
        pyplot.title(
            "Measuring the model's effectiveness for different rejection thresholds\n" +
            f"cost correct: {self.costs.cost_correct}, cost incorrect: {self.costs.cost_incorrect}" +
            f", cost rejection: {self.costs.cost_rejection}")
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

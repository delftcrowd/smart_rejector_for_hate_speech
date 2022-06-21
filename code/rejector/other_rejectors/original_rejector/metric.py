from matplotlib import pyplot
from original_rejector.pdfs import PDFs
from original_rejector.prediction import Prediction
from original_rejector.values import Values
from typing import List, Dict
import numpy as np


class Metric():
    """Provides helper functions for handling predictions.
    """

    def __init__(self, values: Values, predictions: List[Prediction],
                 estimator_conf: Dict[str, Dict[str, object]] = None) -> None:
        """Initializes the metric

        Args:
            values (Values): The values of correct/incorrect predictions and rejections.
            predictions (List[Prediction]): The list of predictions.
            estimator_conf (Dict[str, Dict[str, object]], optional): Dictionary that contains
                the KDE params. Defaults to {} and finds the optimal
                params automatically. Defaults to None.
        """
        self.values = values
        self.predictions = predictions
        self.estimator_conf = estimator_conf
        self.pdfs = PDFs(self.predictions, self.estimator_conf)

    def calculate_effectiveness(self, threshold: float) -> float:
        """Calculates the effectiveness of a model for a specific threshold and value values

        Args:
            threshold (float): The reliability threshold value.

        Returns:
            float: The effectiveness of the model.
        """
        value_correct = self.values.value_correct
        value_incorrect = self.values.value_incorrect
        value_rejection = self.values.value_rejection
        correct = self.pdfs.correct
        incorrect = self.pdfs.incorrect

        return (value_correct + value_rejection) * correct.integral(min=threshold, max=1.0) \
            + (value_rejection - value_incorrect) * incorrect.integral(min=threshold, max=1.0) \
            - (value_rejection + value_correct) * correct.integral(min=0.0, max=threshold) \
            + (value_incorrect - value_rejection) * incorrect.integral(min=0.0, max=threshold)

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
        pyplot.annotate(
            f'Maximum effectiveness: (Threshold: {round(thresholds[index], 4)}, Effectiveness: {round(max_effectiveness, 4)})',
            (thresholds[index],
             max_effectiveness))
        pyplot.xlabel("Rejection threshold (σ)")
        pyplot.ylabel("Effectiveness of the model (P(σ))")
        pyplot.title(
            "Measuring the model's effectiveness for different rejection thresholds\n" +
            f"value correct: {self.values.value_correct}, value incorrect: {self.values.value_incorrect}" +
            f", value rejection: {self.values.value_rejection}")
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

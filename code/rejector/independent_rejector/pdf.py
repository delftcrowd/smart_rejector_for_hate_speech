from __future__ import annotations
import numpy as np
import statsmodels.api as sm
from typing import List
from scipy.integrate import simps
from independent_rejector.prediction import Prediction


class PDF():
    """Class that contains information about the Probability Density Function.
    """

    def __init__(self, predictions: List[Prediction], fraction: float, kde: sm.nonparametric.KDEMultivariate = None) -> None:
        """
        Args:
            predictions (List[Prediction]): The list of Predictions.
            fraction (float): The fraction of the predictions for the total set of predictions.
            kde (sm.nonparametric.KDEMultivariate, optional): The KernelDensity estimator. Default to None.
        """
        if kde != None:
            self.predictions = predictions
            self.pdf_x = np.linspace(0, 1, 10000)
            self.pdf_y = kde.pdf(self.pdf_x)
            self.fraction = fraction
        else:
            self.predictions = predictions
            self.pdf_x = []
            self.pdf_y = []
            self.fraction = fraction

    def D(self, threshold: float) -> float:
        """Returns the Probability Density Function value for a specific threshold.

        Args:
            threshold (float): The threshold for which you want to retrieve the PDF y axis value.

        Returns:
            float: The PDF value
        """
        index = -1
        for idx, pdf_x in enumerate(self.pdf_x):
            if idx < len(self.pdf_x) - 1:
                next_pdf_x = self.pdf_x[idx + 1]
                if pdf_x <= threshold <= next_pdf_x:
                    index = idx
        if index != -1:
            return np.average([self.pdf_y[index], self.pdf_y[index+1]]) * self.fraction
        else:
            return 0

    def integral(self, threshold: float) -> float:
        """Calculate the area under PDF curve for the interval [0, threshold].

        Args:
            threshold (float): The rejection threshold value.

        Returns:
            float: The integral of the PDF for interval [0, threshold]
        """
        below_threshold_filtered = list(
            filter(lambda v: v < threshold, self.pdf_x))
        if len(below_threshold_filtered) == 0:
            return 0
        else:
            return simps(
                self.pdf_y[:len(below_threshold_filtered)], below_threshold_filtered) * self.fraction

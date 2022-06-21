from __future__ import annotations
from original_rejector.prediction import Prediction
from scipy.integrate import simps
from typing import List
import numpy as np
import statsmodels.api as sm


class PDF():
    """Class that contains information about the Probability Density Function.
    """

    def __init__(self, predictions: List[Prediction],
                 fraction: float, kde: sm.nonparametric.KDEMultivariate = None) -> None:
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
        """Returns the Probability Density Function value for a specific threshold

        Args:
            threshold (float): The threshold for which you want to retrieve the PDF y axis value.

        Returns:
            float: the PDF value
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

    def integral(self, min: float = 0.0, max: float = 1.0) -> float:
        """Calculate the area under PDF curve for the interval [min, max].

        Args:
            threshold (float): The rejection threshold value.

        Returns:
            float: The integral of the PDF for interval [min, max]
        """
        zipped = list(zip(self.pdf_x, self.pdf_y))
        filtered = [point for point in zipped if point[0] >= min and point[0] <= max]
        filtered_y = [point[1] for point in filtered]
        filtered_x = [point[0] for point in filtered]
        if len(filtered_y) == 0:
            return 0
        else:
            return simps(filtered_y, filtered_x) * self.fraction

from __future__ import annotations
from typing import List, Dict
from new_idea.prediction import Prediction
import numpy as np
from new_idea.pdf import PDF
import logging
import statsmodels.api as sm


class PDFs():
    """Class that contains information about the Probability Density Functions of correct and incorrect predictions.
    """

    def __init__(self, predictions: List[Prediction], estimator_conf: Dict[str, Dict[str, object]] = None) -> None:
        """Initializes the Probability Density Functions for correct and incorrect predictions.

        Args:
             predictions (List[Prediction]): The list of Predictions.
            estimator_conf (Dict[str, Dict[str, object]], optional): The KDE params. Defaults to None.
                If none were passed, then the optimal params values will be calculated and logged.
        """
        correct = Prediction.set_of_correct(predictions)
        incorrect = Prediction.set_of_incorrect(predictions)

        fraction_correct = len(correct) / len(predictions)
        fraction_incorrect = len(incorrect) / len(predictions)

        logging.info("Fraction correct: %s", fraction_correct)
        logging.info("Fraction incorrect: %s", fraction_incorrect)

        if estimator_conf != None:
            correct_bandwidth = estimator_conf.get('Correct').get('bandwidth')
            incorrect_bandwidth = estimator_conf.get(
                'Incorrect').get('bandwidth')

            self.correct = self.to_pdf(
                correct, fraction_correct, correct_bandwidth)
            self.incorrect = self.to_pdf(
                incorrect, fraction_incorrect, incorrect_bandwidth)
        else:
            self.correct = self.to_pdf(correct, fraction_correct)
            self.incorrect = self.to_pdf(incorrect, fraction_incorrect)

    @staticmethod
    def kde(values: List[float], bandwidth: str | float) -> sm.nonparametric.KDEMultivariate:
        """Returns the best Kernel Density Estimator
        by performing cross validation by trying out different bandwidth
        (smoothing factor) values or uses the user-specified bandwidth

        Args:
            values (List[float]): A list of values that needs to be estimated.
            bandwidth (str | float): user-specified bandwidth for the KDE.

        Returns:
            sm.nonparametric.KDEMultivariate: [description]
        """
        if isinstance(bandwidth, float):
            bw = [bandwidth]
        else:
            bw = bandwidth

        kde = sm.nonparametric.KDEMultivariate(
            data=values, var_type='c', bw=bw)
        logging.info("KDE optimal bandwidths: %s", kde.bw)
        return kde

    @ classmethod
    def estimator(
            cls, predictions: List[Prediction],
            bandwidth: str | float = "cv_ml") -> sm.nonparametric.KDEMultivariate:
        """Returns the KernelDensity estimator that is fitted on the predictions.
        If no bandwidths are passed, then the optimal bandwidths are automatically calculated (is slower).

        Args:
            predictions (List[Prediction]): The list of predictions.
            bandwidth (str | float, optional): The optimal bandwidth for the KDE. Defaults to "cv_ml".

        Returns:
            sm.nonparametric.KDEMultivariate: The KernelDensity estimator fitted on the predictions.
        """
        reliability_values = np.asarray(
            list(map(lambda p: p.predicted_value, predictions)))

        reliability_values = reliability_values.reshape(
            (len(reliability_values), 1))

        return cls.kde(reliability_values, bandwidth)

    @ classmethod
    def to_pdf(cls, predictions: List[Prediction], fraction: float, bandwidth: str | float = "cv_ml") -> PDF:
        """Creates a Probability Density Function object from a list of predictions.

        Args:
            predictions (List[Prediction]): The list of predictions.
            fraction (float): the fraction of predictions (either correct or incorrect) among the total set of predictions.
            bandwidth (str | float, optional): The optimal bandwidth for the KDE. Defaults to "cv_ml".

        Returns:
            PDF: The Probability Density Function for the list of predictions.
        """
        if len(predictions) > 20:
            estimator = cls.estimator(predictions, bandwidth)
            return PDF(predictions, fraction, estimator)
        else:
            logging.warning("Not enough samples for creating the PDF")
            return PDF(predictions, fraction)

from __future__ import annotations
from rejector.pdf import PDF
from rejector.prediction import Prediction
from typing import List, Dict
import logging
import numpy as np
import statsmodels.api as sm


class PDFs:
    """Class that contains information about the Probability Density Functions of TP, TN, FP, and FN."""

    def __init__(
        self,
        predictions: List[Prediction],
        estimator_conf: Dict[str, Dict[str, object]] = None,
    ) -> None:
        """Initializes the Probability Density Functions for TP, TN, FP, and FN.

        Args:
            predictions (List[Prediction]): The list of Predictions.
            estimator_conf (Dict[str, Dict[str, object]], optional): The KDE params. Defaults to None.
                If none were passed, then the optimal params values will be calculated and logged.
        """
        tps = Prediction.set_of_tps(predictions)
        tns = Prediction.set_of_tns(predictions)
        fps = Prediction.set_of_fps(predictions)
        fns = Prediction.set_of_fns(predictions)

        fraction_tps = len(tps) / len(predictions)
        fraction_tns = len(tns) / len(predictions)
        fraction_fps = len(fps) / len(predictions)
        fraction_fns = len(fns) / len(predictions)

        logging.info("Fraction of TPS: %s", fraction_tps)
        logging.info("Fraction of TNS: %s", fraction_tns)
        logging.info("Fraction of FPS: %s", fraction_fps)
        logging.info("Fraction of FNS: %s", fraction_fns)

        if estimator_conf != None:
            tps_bandwidth = estimator_conf.get("TPS").get("bandwidth")
            tns_bandwidth = estimator_conf.get("TNS").get("bandwidth")
            fps_bandwidth = estimator_conf.get("FPS").get("bandwidth")
            fns_bandwidth = estimator_conf.get("FNS").get("bandwidth")

            self.tps = self.to_pdf(tps, fraction_tps, tps_bandwidth)
            self.tns = self.to_pdf(tns, fraction_tns, tns_bandwidth)
            self.fps = self.to_pdf(fps, fraction_fps, fps_bandwidth)
            self.fns = self.to_pdf(fns, fraction_fns, fns_bandwidth)
        else:
            self.tps = self.to_pdf(tps, fraction_tps)
            self.tns = self.to_pdf(tns, fraction_tns)
            self.fps = self.to_pdf(fps, fraction_fps)
            self.fns = self.to_pdf(fns, fraction_fns)

    @staticmethod
    def kde(
        values: List[float], bandwidth: str | float
    ) -> sm.nonparametric.KDEMultivariate:
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

        kde = sm.nonparametric.KDEMultivariate(data=values, var_type="c", bw=bw)
        logging.info("KDE optimal bandwidths: %s", kde.bw)
        return kde

    @classmethod
    def estimator(
        cls, predictions: List[Prediction], bandwidth: str | float = "cv_ml"
    ) -> sm.nonparametric.KDEMultivariate:
        """Returns the KernelDensity estimator that is fitted on the predictions.
        If no bandwidths are passed, then the optimal bandwidths are automatically calculated (is slower).

        Args:
            predictions (List[Prediction]): The list of predictions.
            bandwidth (str | float, optional): The optimal bandwidth for the KDE. Defaults to "cv_ml".

        Returns:
            sm.nonparametric.KDEMultivariate: The KernelDensity estimator fitted on the predictions.
        """
        reliability_values = np.asarray(
            list(map(lambda p: p.predicted_value, predictions))
        )

        reliability_values = reliability_values.reshape((len(reliability_values), 1))

        return cls.kde(reliability_values, bandwidth)

    @classmethod
    def to_pdf(
        cls,
        predictions: List[Prediction],
        fraction: float,
        bandwidth: str | float = "cv_ml",
    ) -> PDF:
        """Creates a Probability Density Function object from a list of predictions.

        Args:
            predictions (List[Prediction]): The list of predictions.
            fraction (float): the fraction of predictions (either TP, TN, FP, or FN) among the total set of predictions.
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

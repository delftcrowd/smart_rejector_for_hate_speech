from rejector.values import Values
from rejector.metric import Metric
from rejector.prediction import Prediction
import numpy as np
import unittest


class TestMetric(unittest.TestCase):
    def setUp(self):
        predictions = []
        # Add False Negatives
        for p in np.linspace(0, 0.3, 200):
            prediction = Prediction("negative", "positive", p, "positive", None)
            predictions.append(prediction)

        # Add False Positives
        for p in np.linspace(0, 0.3, 200):
            prediction = Prediction("positive", "negative", p, "positive", None)
            predictions.append(prediction)

        # Add True Negatives
        for p in np.linspace(0.7, 1.0, 200):
            prediction = Prediction("negative", "negative", p, "positive", None)
            predictions.append(prediction)

        # Add True Positives
        for p in np.linspace(0.7, 1.0, 200):
            prediction = Prediction("positive", "positive", p, "positive", None)
            predictions.append(prediction)

        self.predictions = predictions

    def test_ideal_case(self):
        values = Values(
            value_TP=1, value_TN=1, value_FP=5, value_FN=5, value_rejection=2
        )
        metric = Metric(values, self.predictions)

        # The effectiveness should be optimal for threshold 0.5 since we accept all correct
        # and reject all incorrect predictions.
        self.assertTrue(2.9 <= metric.calculate_effectiveness(0.5) <= 3.0)

        # The effectiveness should be around 0 when all predictions are accepted.
        self.assertTrue(0 <= metric.calculate_effectiveness(0) <= 0.1)

        # The effectiveness should be around 0 when all predictions are rejected.
        self.assertAlmostEqual(metric.calculate_effectiveness(1.0), 0.0, 2)

    def test_ideal_case_with_kde_config(self):
        values = Values(
            value_TP=1, value_TN=1, value_FP=5, value_FN=5, value_rejection=2
        )
        estimator_conf = {
            "TPS": {"bandwidth": 0.02018681},
            "TNS": {"bandwidth": 0.02018681},
            "FPS": {"bandwidth": 0.02018681},
            "FNS": {"bandwidth": 0.02018681},
        }
        metric = Metric(values, self.predictions, estimator_conf)

        # The effectiveness should be optimal for threshold 0.5 since we accept all correct
        # and reject all incorrect predictions.
        self.assertTrue(2.9 <= metric.calculate_effectiveness(0.5) <= 3.0)

        # The effectiveness should be around 0 when all predictions are accepted.
        self.assertTrue(0 <= metric.calculate_effectiveness(0) <= 0.1)

        # The effectiveness should be around 0 when all predictions are rejected.
        self.assertAlmostEqual(metric.calculate_effectiveness(1.0), 0.0, 2)


if __name__ == "__main__":
    unittest.main()

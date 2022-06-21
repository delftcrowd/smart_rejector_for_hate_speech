from original_rejector.metric import Metric
from original_rejector.prediction import Prediction
from original_rejector.values import Values
import numpy as np
import unittest


class TestMetric(unittest.TestCase):
    def setUp(self):
        predictions = []
        # Add False Negatives
        for p in np.linspace(0, 0.3, 200):
            prediction = Prediction(
                "negative", "positive", p, None)
            predictions.append(prediction)

        # Add False Positives
        for p in np.linspace(0, 0.3, 200):
            prediction = Prediction(
                "positive", "negative", p, None)
            predictions.append(prediction)

        # Add True Negatives
        for p in np.linspace(0.7, 1.0, 200):
            prediction = Prediction(
                "negative", "negative", p, None)
            predictions.append(prediction)

        # Add True Positives
        for p in np.linspace(0.7, 1.0, 200):
            prediction = Prediction(
                "positive", "positive", p, None)
            predictions.append(prediction)

        self.predictions = predictions

    def test_ideal_case(self):
        values = Values(value_correct=1, value_incorrect=5, value_rejection=2)
        metric = Metric(values, self.predictions)

        # The effectiveness should be optimal for threshold 0.5 since we accept all correct
        # and reject all incorrect predictions.
        self.assertTrue(2.9 <= metric.calculate_effectiveness(0.5) <= 3.0)

        # The effectiveness should be around 0 when all predictions are accepted.
        self.assertTrue(0 <= metric.calculate_effectiveness(0) <= 0.1)

        # The effectiveness should be around 0 when all predictions are rejected.
        self.assertAlmostEqual(metric.calculate_effectiveness(1.0), 0.0, 2)


    def test_ideal_case_with_kde_config(self):
        values = Values(value_correct=1, value_incorrect=5, value_rejection=2)
        estimator_conf = {'Correct': {'bandwidth': 0.02018681},
                          'Incorrect': {'bandwidth': 0.02018681}}
        metric = Metric(values, self.predictions, estimator_conf)

        # The effectiveness should be optimal for threshold 0.5 since we accept all correct
        # and reject all incorrect predictions.
        self.assertTrue(2.9 <= metric.calculate_effectiveness(0.5) <= 3.0)

        # The effectiveness should be around 0 when all predictions are accepted.
        self.assertTrue(0 <= metric.calculate_effectiveness(0) <= 0.1)

        # The effectiveness should be around 0 when all predictions are rejected.
        self.assertAlmostEqual(metric.calculate_effectiveness(1.0), 0.0, 2)


if __name__ == '__main__':
    unittest.main()

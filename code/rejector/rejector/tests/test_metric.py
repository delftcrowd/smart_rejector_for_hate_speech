import unittest
import numpy as np
from rejector.costs import Costs
from rejector.metric import Metric
from rejector.prediction import Prediction


class TestMetric(unittest.TestCase):
    def setUp(self):
        predictions = []
        # Add False Negatives
        for p in np.linspace(0, 0.3, 100):
            prediction = Prediction(
                "negative", "positive", p, "positive", None)
            predictions.append(prediction)

        # Add False Positives
        for p in np.linspace(0, 0.3, 100):
            prediction = Prediction(
                "positive", "negative", p, "positive", None)
            predictions.append(prediction)

        # Add True Negatives
        for p in np.linspace(0.7, 1.0, 100):
            prediction = Prediction(
                "negative", "negative", p, "positive", None)
            predictions.append(prediction)

        # Add True Positives
        for p in np.linspace(0.7, 1.0, 100):
            prediction = Prediction(
                "positive", "positive", p, "positive", None)
            predictions.append(prediction)

        self.predictions = predictions

    def test_ideal_case(self):
        costs = Costs(1, 1, 5, 5, 1)
        metric = Metric(costs, self.predictions)

        # The effectiveness of the reject option should be around 1.0  when all
        # samples are rejected
        self.assertTrue(0.90 <= metric.calculate_effectiveness(1.0) <= 1.1)

        # The effectiveness of the reject option should be around 2.0
        # The actual value is lower since Kernel Density Estimation cannot perfectly
        # mimic the square shape of all predictions falling into either the 0-0.5 or 0.5-1.0 range.
        self.assertTrue(1.90 <= metric.calculate_effectiveness(0.5) <= 2.1)

        # The effectiveness of the reject option should be around 0 when all predictions
        # are accepted.
        self.assertTrue(0 <= metric.calculate_effectiveness(0) <= 0.2)

    def test_ideal_case_with_kde_config(self):
        costs = Costs(1, 1, 5, 5, 1)
        estimator_conf = {'TPS': {'bandwidth': 0.02018681}, 'TNS': {'bandwidth': 0.02018681}, 'FPS': {
            'bandwidth': 0.02018681}, 'FNS': {'bandwidth': 0.02018681}}
        metric = Metric(costs, self.predictions, estimator_conf)

        # The effectiveness of the reject option should be around 1.0  when all
        # samples are rejected
        self.assertTrue(0.90 <= metric.calculate_effectiveness(1.0) <= 1.1)

        # The effectiveness of the reject option should be around 2.0
        # The actual value is lower since Kernel Density Estimation cannot perfectly
        # mimic the square shape of all predictions falling into either the 0-0.5 or 0.5-1.0 range.
        self.assertTrue(1.90 <= metric.calculate_effectiveness(0.5) <= 2.1)

        # The effectiveness of the reject option should be around 0 when all predictions
        # are accepted.
        self.assertTrue(0 <= metric.calculate_effectiveness(0) <= 0.2)


if __name__ == '__main__':
    unittest.main()

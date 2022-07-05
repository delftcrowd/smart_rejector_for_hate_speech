from rejector.prediction import Prediction
import os
import pickle
import unittest


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.tp = Prediction("positive", "positive", 0.8, "positive", None)
        self.tn = Prediction("negative", "negative", 0.8, "positive", None)
        self.fp = Prediction("positive", "negative", 0.8, "positive", None)
        self.fn = Prediction("negative", "positive", 0.8, "positive", None)
        self.predictions = [self.tp, self.tn, self.fp, self.fn]

    def test_set_of_tps(self):
        self.assertEqual(
            Prediction.set_of_tps(self.predictions),
            [Prediction("positive", "positive", 0.8, "positive", None)],
        )

    def test_set_of_tns(self):
        self.assertEqual(
            Prediction.set_of_tns(self.predictions),
            [Prediction("negative", "negative", 0.8, "positive", None)],
        )

    def test_set_of_fps(self):
        self.assertEqual(
            Prediction.set_of_fps(self.predictions),
            [Prediction("positive", "negative", 0.8, "positive", None)],
        )

    def test_set_of_fns(self):
        self.assertEqual(
            Prediction.set_of_fns(self.predictions),
            [Prediction("negative", "positive", 0.8, "positive", None)],
        )

    def test_count_above_threshold(self):
        self.assertEqual(Prediction.count_above_threshold(self.predictions, 0.5), 4)
        self.assertEqual(Prediction.count_above_threshold(self.predictions, 0.7), 4)
        self.assertEqual(Prediction.count_above_threshold(self.predictions, 0.8), 0)

    def test_count_below_threshold(self):
        self.assertEqual(Prediction.count_below_threshold(self.predictions, 0.5), 0)
        self.assertEqual(Prediction.count_below_threshold(self.predictions, 0.7), 0)
        self.assertEqual(Prediction.count_below_threshold(self.predictions, 0.8), 4)

    def test_is_tp(self):
        self.assertTrue(self.tp.is_tp())
        self.assertFalse(self.tn.is_tp())
        self.assertFalse(self.fp.is_tp())
        self.assertFalse(self.fn.is_tp())

    def test_is_tn(self):
        self.assertFalse(self.tp.is_tn())
        self.assertTrue(self.tn.is_tn())
        self.assertFalse(self.fp.is_tn())
        self.assertFalse(self.fn.is_tn())

    def test_is_fp(self):
        self.assertFalse(self.tp.is_fp())
        self.assertFalse(self.tn.is_fp())
        self.assertTrue(self.fp.is_fp())
        self.assertFalse(self.fn.is_fp())

    def test_is_fn(self):
        self.assertFalse(self.tp.is_fn())
        self.assertFalse(self.tn.is_fn())
        self.assertFalse(self.fp.is_fn())
        self.assertTrue(self.fn.is_fn())

    def test_load_predictions(self):
        dict_predictions = [
            {
                "predicted_class": "positive",
                "actual_class": "positive",
                "predicted_value": 0.8,
                "text": None,
            },
            {
                "predicted_class": "negative",
                "actual_class": "negative",
                "predicted_value": 0.8,
                "text": None,
            },
        ]
        file = open("temp_predictions.p", "wb")
        pickle.dump(dict_predictions, file)
        file.close()
        loaded_predictions = Prediction.load("temp_predictions.p", "positive")
        self.assertListEqual(loaded_predictions, [self.tp, self.tn])
        os.remove("temp_predictions.p")


if __name__ == "__main__":
    unittest.main()

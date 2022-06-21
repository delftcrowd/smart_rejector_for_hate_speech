from original_rejector.prediction import Prediction
import os
import pickle
import unittest


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.tp = Prediction("positive", "positive", 0.8, None)
        self.tn = Prediction("negative", "negative", 0.8, None)
        self.fp = Prediction("positive", "negative", 0.8, None)
        self.fn = Prediction("negative", "positive", 0.8, None)
        self.predictions = [self.tp, self.tn, self.fp, self.fn]

    def test_set_of_correct(self):
        self.assertEqual(Prediction.set_of_correct(self.predictions), [
            Prediction("positive", "positive", 0.8, None),
            Prediction("negative", "negative", 0.8, None)])

    def test_set_of_incorrect(self):
        self.assertEqual(Prediction.set_of_incorrect(self.predictions), [
            Prediction("positive", "negative", 0.8, None),
            Prediction("negative", "positive", 0.8, None)])

    def test_is_correct(self):
        self.assertTrue(self.tp.is_correct())
        self.assertTrue(self.tn.is_correct())
        self.assertFalse(self.fp.is_correct())
        self.assertFalse(self.fn.is_correct())

    def test_load_predictions(self):
        dict_predictions = [{"predicted_class": "positive",
                             "actual_class": "positive", "predicted_value": 0.8, "text": None},
                            {"predicted_class": "negative",
                             "actual_class": "negative", "predicted_value": 0.8, "text": None}]
        file = open("temp_predictions.p", "wb")
        pickle.dump(dict_predictions, file)
        file.close()
        loaded_predictions = Prediction.load("temp_predictions.p")
        self.assertListEqual(loaded_predictions, [self.tp, self.tn])
        os.remove('temp_predictions.p')


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations
import pickle
from typing import List


class Prediction:
    """Contains information about the prediction"""

    def __init__(
        self,
        predicted_class: str,
        actual_class: str,
        predicted_value: float,
        gold_class: str,
        text: str,
    ) -> None:
        """
        Args:
            predicted_class (str): the predicted class name
            actual_class (str): the actual class name
            predicted_value (float): the predicted value or confidence value
            gold_class (str): the gold class name used for determining if something is a TP, TN, FP, or FN
            text (str): the original text message
        """
        self.predicted_class = predicted_class
        self.actual_class = actual_class
        self.predicted_value = predicted_value
        self.gold_class = gold_class
        self.text = text

    @staticmethod
    def set_of_tps(predictions: List[Prediction]) -> List[Prediction]:
        """Returns the set of True Positives

        Args:
            predictions (List[Prediction]): list of predictions

        Returns:
            List[Prediction]: list of TP predictions
        """
        return list(filter(lambda p: p.is_tp(), predictions))

    @staticmethod
    def set_of_tns(predictions: List[Prediction]) -> List[Prediction]:
        """Returns the set of True Negatives

        Args:
            predictions (List[Prediction]): list of predictions

        Returns:
            List[Prediction]: list of TN predictions
        """
        return list(filter(lambda p: p.is_tn(), predictions))

    @staticmethod
    def set_of_fps(predictions: List[Prediction]) -> List[Prediction]:
        """Returns the set of False Positives

        Args:
            predictions (List[Prediction]): list of predictions

        Returns:
            List[Prediction]: list of FP predictions
        """
        return list(filter(lambda p: p.is_fp(), predictions))

    @staticmethod
    def set_of_fns(predictions: List[Prediction]) -> List[Prediction]:
        """Returns the set of False Negatives

        Args:
            predictions (List[Prediction]): list of predictions

        Returns:
            List[Prediction]: list of FN predictions
        """
        return list(filter(lambda p: p.is_fn(), predictions))

    @staticmethod
    def count_above_threshold(predictions: List[Prediction], threshold: float) -> int:
        """Counts the number of predictions with a confidence value larger than `threshold`.

        Args:
            predictions (List[Prediction]): list of predictions.
            threshold (float): the confidence threshold.

        Returns:
            int: number of predictions.
        """
        return len(list(filter(lambda p: p.predicted_value > threshold, predictions)))

    @staticmethod
    def count_below_threshold(predictions: List[Prediction], threshold: float) -> int:
        """Counts the number of predictions with a confidence value less than or equal to `threshold`.

        Args:
            predictions (List[Prediction]): list of predictions.
            threshold (float): the confidence threshold.

        Returns:
            int: number of predictions.
        """
        return len(list(filter(lambda p: p.predicted_value <= threshold, predictions)))

    def is_tp(self) -> bool:
        """Checks if a prediction is a True Positive

        Returns:
            bool
        """
        return (
            self.predicted_class == self.gold_class
            and self.predicted_class == self.actual_class
        )

    def is_tn(self) -> bool:
        """Checks if a prediction is a True Negative

        Returns:
            bool
        """
        return (
            self.predicted_class != self.gold_class
            and self.predicted_class == self.actual_class
        )

    def is_fp(self):
        """Checks if a prediction is a False Positve

        Returns:
            bool
        """
        return (
            self.predicted_class == self.gold_class
            and self.predicted_class != self.actual_class
        )

    def is_fn(self):
        """Checks if a prediction is a False Negative

        Returns:
            bool
        """
        return (
            self.actual_class == self.gold_class
            and self.predicted_class != self.actual_class
        )

    @staticmethod
    def load(path: str, gold_class: str) -> List[Prediction]:
        """Loads a list of predictions from a file
        The file should be a pickle file containing a list of dictionaries with the following attributes:

        Args:
            path (str): the path of the file containing the predictions
            gold_class (str): the gold class name used for determining if something is a TP, TN, FP, or FN

        Returns:
            List[Prediction]: the list of predictions
        """
        file = open(path, "rb")
        predictions = pickle.load(file)
        file.close()
        return list(
            map(
                lambda res: Prediction(
                    res["predicted_class"],
                    res["actual_class"],
                    res["predicted_value"],
                    gold_class,
                    res["text"],
                ),
                predictions,
            )
        )

    def __eq__(self, other):
        return (
            self.predicted_class == other.predicted_class
            and self.actual_class == other.actual_class
            and self.gold_class == other.gold_class
            and self.text == other.text
        )

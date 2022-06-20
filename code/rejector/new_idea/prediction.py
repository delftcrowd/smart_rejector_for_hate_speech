from __future__ import annotations
import pickle
from typing import List


class Prediction():
    """Contains information about the prediction
    """

    def __init__(self, predicted_class: str, actual_class: str, predicted_value: float, text: str) -> None:
        """
        Args:
            predicted_class (str): the predicted class name
            actual_class (str): the actual class name
            predicted_value (float): the predicted value or confidence value
            text (str): the original text
        """
        self.predicted_class = predicted_class
        self.actual_class = actual_class
        self.predicted_value = predicted_value
        self.text = text

    @staticmethod
    def set_of_correct(predictions: List[Prediction]) -> List[Prediction]:
        """Returns the set of correct predictions

        Args:
            predictions (List[Prediction]): list of predictions

        Returns:
            List[Prediction]: list of correct predictions
        """
        return list(filter(lambda p: p.is_correct(), predictions))

    @staticmethod
    def set_of_incorrect(predictions: List[Prediction]) -> List[Prediction]:
        """Returns the set of incorrect predictions

        Args:
            predictions (List[Prediction]): list of predictions

        Returns:
            List[Prediction]: list of incorrect predictions
        """
        return list(filter(lambda p: not p.is_correct(), predictions))

    def is_correct(self) -> bool:
        """Checks if a prediction is correct

        Returns:
            bool
        """
        return self.predicted_class == self.actual_class

    @staticmethod
    def load(path: str) -> List[Prediction]:
        """Loads a list of predictions from a file
        The file should be a pickle file containing a list of dictionaries with the following attributes:

        Args:
            path (str): the path of the file containing the predictions

        Returns:
            List[Prediction]: the list of predictions
        """
        file = open(path, "rb")
        predictions = pickle.load(file)
        file.close()
        return list(map(lambda res: Prediction(res['predicted_class'], res['actual_class'], res['predicted_value'], res['text']), predictions))

    def __eq__(self, other):
        return self.predicted_class == other.predicted_class and \
            self.actual_class == other.actual_class and \
            self.text == other.text

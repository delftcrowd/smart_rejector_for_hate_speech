from __future__ import annotations
from sklearn.model_selection import train_test_split

import pickle
import numpy as np


class Reader:
    """Helper class for reading and splitting the text data."""

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def split(self, X: list, y: list) -> tuple[list, list, list, list]:
        """Splits a list of data samples X and list of labels Y into a train and
        test dataset.

        Args:
            X (list): list of data samples.
            y (list): list of labels.

        Returns:
            tuple[list, list, list, list]: splitted train and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=10, stratify=y, test_size=0.10
        )
        return X_train, X_test, y_train, y_test

    def split_with_validation(
        self, X: list, y: list
    ) -> tuple[list, list, list, list, list, list]:
        """Splits a list of data samples X and list of labels Y into a train,
        test, and validation dataset.

        Args:
            X (list): list of data samples.
            y (list): list of labels.

        Returns:
            tuple[list, list, list, list]: splitted train, validation, and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=10, stratify=y, test_size=0.10
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, random_state=10, test_size=0.2
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def load(self) -> tuple[list, list]:
        """Loads the data from an external file.

        Returns:
            tuple[list, list]: list of data samples and a list of labels.
        """
        data = pickle.load(open(self.filename, "rb"))
        X = []
        y = []
        for i in range(len(data)):
            X.append(data[i]["text"])
            y.append(data[i]["label"])

        return X, y

from __future__ import annotations
from sklearn.model_selection import train_test_split

import pickle
import preprocessor as p
import html
import regex as re
from wordsegment import load, segment

load()


class Reader:
    """Helper class for reading, preprocessing, and splitting the text data."""

    @staticmethod
    def split(X: list, y: list) -> tuple[list, list, list, list]:
        """Splits a list of data samples X and list of labels Y into a train and
        test dataset.

        Args:
            X (list): list of data samples.
            y (list): list of labels.

        Returns:
            tuple[list, list, list, list]: splitted train and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=10, stratify=y, test_size=0.2
        )
        return X_train, X_test, y_train, y_test

    @classmethod
    def preprocess(cls, X: list) -> list:
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)

        return list(
            map(lambda text: p.tokenize(html.unescape(cls.split_hashtags(text))), X)
        )

    @staticmethod
    def split_with_validation(
        X: list, y: list
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
            X, y, random_state=10, stratify=y, test_size=0.2
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, random_state=10, test_size=0.25
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def load(filename: str) -> tuple[list, list]:
        """Loads the data from an external file.

        Returns:
            tuple[list, list]: list of data samples and a list of labels.
        """
        data = pickle.load(open(filename, "rb"))
        X = []
        y = []
        for i in range(len(data)):
            X.append(data[i]["text"])
            y.append(data[i]["label"])

        return X, y

    @staticmethod
    def split_hashtags(text: str) -> str:
        hashtags = re.findall(r"(#\w+)", text)
        for hashtag in hashtags:
            hashtag_words = " ".join(segment(hashtag.lstrip("#")))
            text = text.replace(hashtag, hashtag_words)
        return text

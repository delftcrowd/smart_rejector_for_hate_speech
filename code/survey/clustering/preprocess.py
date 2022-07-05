from typing import List
import csv
import html
import logging
import preprocessor as p
import pickle


class Preprocess:
    """Class that contains helper functions for preprocessing the data."""

    @classmethod
    def filter_waseem_hovy(cls, X: list, label: str) -> list:
        """Filters the pickle file with the Waseem and Hovy data based on the labels.

        Args:
            X (list): input Waseem and Hovy data list that needs to be filtered.
            label (str): filter on 'hate' or 'none'

        Returns:
            list: filtered Waseem and Hovy dataset.
        """
        logging.info("Original data length: %s", len(X) - 1)

        # Filter out
        filtered_data = list(filter(lambda x: x["label"] == label, X))

        logging.info("After applying filters: %s", len(filtered_data))

        filtered_data = [x for x in filtered_data if cls.valid_text(x["text"])]

        logging.info(
            "Data length after removing invalid tweets: %s", len(filtered_data)
        )

        # Remove all tweets that are invalid (contain urls, mentions, or not enough text after cleaning)
        return filtered_data

    @classmethod
    def filter_sem_eval(cls, X: list, HS: str, TR: str, AG: str) -> list:
        """Filters the CSV data of the SemEval 2019 dataset based on the column values.

        Args:
            X (list): input SemEval data list that needs to be filtered.
            HS (str): filter on hateful '1' or non-hateful '0' tweets.
            TR (str): filter on individually targeted '1' or generic group '0' targeted tweets.
            AG (str): filter on aggressive '1' or non-aggressive '0' tweets

        Returns:
            list: filtered SemEval dataset.
        """
        logging.info("Original data length: %s", len(X) - 1)

        # Filter out
        filtered_data = list(
            filter(lambda x: x[2] == HS and x[3] == TR and x[4] == AG, X)
        )

        logging.info("After applying filters: %s", len(filtered_data))

        # Remove first row since these contains headers
        filtered_data = filtered_data[1:]
        filtered_data = [x for x in filtered_data if cls.valid_text(x[1])]

        logging.info(
            "Data length after removing invalid tweets: %s", len(filtered_data)
        )

        # Remove all tweets that are invalid (contain urls, mentions, or not enough text after cleaning)
        return filtered_data

    @staticmethod
    def open_pickle(file_path: str) -> list:
        """Opens the pickle file and returns list of rows read.

        Args:
            file_path (str): the file path.

        Returns:
            list: list of rows read.
        """
        file = open(file_path, "rb")
        data = pickle.load(file)
        file.close()
        return data

    @staticmethod
    def open_csv(file_path: str) -> list:
        """Opens the CSV file and returns list of rows read.

        Args:
            file_path (str): the file path.

        Returns:
            list: list of rows read.
        """
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
        return data

    @staticmethod
    def clean(X: List[str]) -> list:
        """Remove html attributes and clean tweets by removing hashtags, mentions, and urls.

        Args:
            X (list): input data list that needs to be cleaned.

        Returns:
            list: list of filtered data.
        """
        X = list(map(lambda x: p.clean(html.unescape(x)), X))

        return X

    @staticmethod
    def valid_text(text: str) -> bool:
        """Checks if the text is valid.
        Valid texts do not contains mentions and urls since the context is often unclear for these tweets,
        and should be empty after cleaning the text up (removing hashtags, urls, mentions and html attributes)

        Args:
            text (str): the string that needs to be checked.

        Returns:
            bool: whether the text is valid or not.
        """
        tokenized_text = p.tokenize(text)
        cleaned_text = p.clean(html.unescape(text))
        return (
            "$MENTION$" not in tokenized_text
            and "$URL$" not in tokenized_text
            and cleaned_text != ""
        )

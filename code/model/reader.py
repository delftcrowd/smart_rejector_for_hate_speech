
from __future__ import annotations
from sklearn.model_selection import train_test_split

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Reader:
    """Helper class for reading, preprocessing and splitting the text data.
    """

    def __init__(self, filename: str, num_classes: int, vocab_len: int) -> None:
        self.filename = filename
        self.num_classes = num_classes
        self.vocab_len = vocab_len

    def tokenizer(self, texts: list) -> Tokenizer:
        """Returns a Keras tokenizer based on the texts input.

        Args:
            texts (list): list of text data.

        Returns:
            Tokenizer: the fitted tokenizer.
        """
        unknown_token = "<OOV>"
        tokenizer = Tokenizer(num_words=self.vocab_len,
                              oov_token=unknown_token)
        tokenizer.fit_on_texts(texts)
        return tokenizer

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
            X, y, random_state=42, stratify=y, test_size=0.10)
        return X_train, X_test, y_train, y_test

    def preprocess_test(self, X: list, y: list, max_len: int) -> tuple[list, list]:
        """Preprocesses a test dataset only.

        Args:
            X (list): data samples.
            y (list): labels.

        Returns:
            tuple[list, list]: preprocessed data samples and labels.
        """

        return X, y

    def preprocess(self, X_train: list, X_test: list, y_train: list, y_test: list, max_len: int = None, tokenizer=None) -> tuple[list, list, list, list, int]:
        """Preprocesses a splitted train and test dataset.

        Args:
            X_train (list): train data samples.
            X_test (list): test data samples.
            y_train (list): train labels.
            y_test (list): test labels.

        Returns:
            tuple[list, list, list, list, int]: preprocessed train and test sets, and the maximum length of a data sample.
        """
        if max_len == None:
            lengths = np.array([len(x.split(" ")) for x in X_train])
            max_len = max(lengths)
        if tokenizer == None:
            tokenizer = self.tokenizer(X_train)
        X_train_sequences = tokenizer.texts_to_sequences(X_train)
        X_test_sequences = tokenizer.texts_to_sequences(X_test)

        X_train = pad_sequences(X_train_sequences, maxlen=max_len,
                                padding='post', truncating='post')
        X_test = pad_sequences(X_test_sequences, maxlen=max_len,
                               padding='post', truncating='post')

        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)

        return X_train, X_test, y_train, y_test, max_len, tokenizer

    def load(self) -> tuple[list, list]:
        """Loads the data from an external file.

        Returns:
            tuple[list, list]: list of data samples and a list of labels.
        """
        data = pickle.load(open(self.filename, 'rb'))
        X = []
        y = []
        for i in range(len(data)):
            X.append(data[i]['text'])
            y.append(data[i]['label'])

        return X, y

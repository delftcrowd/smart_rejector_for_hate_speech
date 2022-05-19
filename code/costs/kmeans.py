from typing import List
from langcodes import Any
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import csv
import preprocessor as p
import html
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging


class KMeansClustering:
    """K-means clustering class.
    """

    def __init__(self):
        """Initializes K-means clustering
        Implementation is based on:
        https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering
        """
        self.K = None
        self.data = []
        self.filtered_data = []
        self.vectorizer = None
        self.svd = None
        self.km = None
        self.predictions = []
        self.distances = []

    @staticmethod
    def most_representative_sample_indices(distances: List[List[float]], predictions: List[int], cluster_index: int, num_samples: int) -> List[int]:
        """Returns the indices of the most representative samples.

        Args:
            distances (List[List[float]]): list of distances between samples and all clusters.
            cluster_index (int): cluster index for which the most representative samples need to be returned.
            num_samples (int): number of most representative samples to return.

        Returns:
            List[int]: list of indices of the most representative samples for cluster.
        """
        cluster_distances = []
        for i, s in enumerate(distances):
            if predictions[i] == cluster_index:
                cluster_distances.append(s[cluster_index])
            else:
                cluster_distances.append(sys.float_info.max)
        sorted_distances_ind = np.argsort(np.array(cluster_distances))

        return sorted_distances_ind[0:num_samples]

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
        return "$MENTION$" not in tokenized_text and "$URL$" not in tokenized_text and cleaned_text != ''

    def open(self, file_path: str) -> list:
        """Opens the file and returns list of rows read.

        Args:
            file_path (str): the file path.

        Returns:
            list: list of rows read.
        """
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
        self.data = data

    def filter(self, HS: str, TR: str, AG: str) -> list:
        """Filters the data of the SemEval 2019 dataset based on the column values.

        Args:
            lines (_type_): _description_
            HS (_type_): _description_
            TR (_type_): _description_
            AG (_type_): _description_
        """
        logging.info("Original data length: ", len(self.data) - 1)

        # Filter out
        filtered_data = list(
            filter(lambda x: x[2] == HS and x[3] == TR and x[4] == AG, self.data))

        logging.info("After applying filters: ", len(filtered_data))

        # Remove first row since these contains headers
        filtered_data = filtered_data[1:]
        self.filtered_data = [
            x for x in filtered_data if self.valid_text(x[1])]

        logging.info("Data length after removing invalid tweets: ",
                     len(self.filtered_data))

        # Remove all tweets that are invalid (contain urls, mentions, or not enough text after cleaning)
        return self.filtered_data

    def clean(self) -> list:
        """Remove html attributes and clean tweets by removing hashtags, mentions, and urls.

        Returns:
            list: list of filtered data.
        """
        X = list(map(lambda x: p.clean(
            html.unescape(x[1])), self.filtered_data))

        return X

    def fit_tfidf(self, X: list) -> list:
        """Fit and transform the data using TF-IDF.

        Returns:
            list: fitted and transformed data with TF-IDF.
        """
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(X)
        self.vectorizer = vectorizer
        return X

    def fit_lsa(self, X: list) -> list:
        """Perform Latent Semantic Analysis (LSA) to reduce dimensions.

        This helps to remove noise. We use Singular Value Decomposition (SVD) on the TF-IDF data,
        which is then known as LSA. For LSA, a dimensionality of 100 is recommended.

        Args:
            X (list): list of tf-idf vectors.

        Returns:
            list: fitted and transformed data with LSA.
        """
        svd = TruncatedSVD(n_components=100)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        explained_variance = svd.explained_variance_ratio_.sum()
        logging.info("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        self.svd = svd
        return X

    def plot_elbow_curve(self, max_k: int, X: list) -> None:
        """Plots elbow curve for all clusters until max_k.

        Args:
            max_k (int): the maximum cluster size to test.
            X (list): list of fitted and transformed LSA .
        """
        plot_x = []
        for k in range(2, max_k):
            km = KMeans(n_clusters=k, init="k-means++")
            plot_x.append(km.inertia_)

        plt.plot(range(2, max_k), plot_x)
        plt.grid(True)
        plt.title('Elbow curve')
        plt.show()

    def cluster(self, X, K) -> Any:
        """Fits and predicts for each sample the nearest cluster.

        Returns:
            Any: the KMeans clustering object.
            X (list): list of fitted and transformed LSA .
        """
        km = KMeans(n_clusters=K, init="k-means++")
        km.fit_predict(X)

        # Calculate distances between samples and all clusters
        self.predictions = km.fit_predict(X)
        self.distances = km.transform(X)
        self.km = km
        self.K = K

        return km

    def print_top_terms(self) -> None:
        """Prints the 10 top terms per cluster.
        """
        original_space_centroids = self.svd.inverse_transform(
            self.km.cluster_centers_)
        ordered_centroids = original_space_centroids.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names_out()

        for k in range(self.K):
            print("=======================================")
            print("Cluster ", k)
            print("Top terms:")
            for i in ordered_centroids[k, :10]:
                print(" %s" % terms[i], end="")
            print()
            print()

    def print_most_representative_samples(self, X, num_samples) -> None:
        """Prints the most representative samples per cluster.
        """
        for k in range(self.K):
            ind = self.most_representative_sample_indices(
                self.distances, self.predictions, k, num_samples)

            print("=======================================")
            print("Cluster ", k)
            print("Most representative sample indices:", ind)
            for i in ind:
                print(X[i])
                print(self.filtered_data[i])
                print()
            print()

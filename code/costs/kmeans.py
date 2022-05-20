from langcodes import Any
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from typing import List
import csv
import html
import logging
import matplotlib.pyplot as plt
import numpy as np
import preprocessor as p
import sys


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
            HS (str): filter on hateful '1' or non-hateful '0' tweets.
            TR (str): filter on individually targeted '1' or generic group '0' targeted tweets.
            AG (str): filter on aggressive '1' or non-aggressive '0' tweets
        """
        logging.info("Original data length: %s", len(self.data) - 1)

        # Filter out
        filtered_data = list(
            filter(lambda x: x[2] == HS and x[3] == TR and x[4] == AG, self.data))

        logging.info("After applying filters: %s", len(filtered_data))

        # Remove first row since these contains headers
        filtered_data = filtered_data[1:]
        self.filtered_data = [
            x for x in filtered_data if self.valid_text(x[1])]

        logging.info("Data length after removing invalid tweets: %s",
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

    def fit_pca(self, X: list, n_components: int = 2) -> list:
        """Perform Principal component analysis (PCA) to reduce dimensions.

        Useful for visualization.

        Args:
            X (list): input data list.
            n_components (int, optional): the dimensionality of the output. Defaults to 2.

        Returns:
            list: fitted and transformed data with LSA.
        """
        pca = PCA(n_components=n_components)

        X = pca.fit_transform(X.toarray())

        self.pca = pca
        return X

    def fit_lsa(self, X: list, n_components: int = 100) -> list:
        """Perform Latent Semantic Analysis (LSA) to reduce dimensions.

        This helps to remove noise. We use Singular Value Decomposition (SVD) on the TF-IDF data,
        which is then known as LSA. For LSA, a dimensionality of 100 is recommended.

        Args:
            X (list): input data list.
            n_components (int, optional): the dimensionality of the output. Defaults to 100.

        Returns:
            list: fitted and transformed data with LSA.
        """
        svd = TruncatedSVD(n_components=n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        explained_variance = svd.explained_variance_ratio_.sum()
        logging.info("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        self.svd = svd
        return X

    def cluster(self, X: list, K: int) -> Any:
        """Fits and predicts for each sample the nearest cluster.

        Args:
            X (list): input data list.
            K (int): number of clusters.

        Returns:
            Any: the KMeans clustering object.
        """
        km = KMeans(n_clusters=K, init="k-means++")
        km.fit(X)

        # Calculate distances between samples and all clusters
        self.km = km
        self.K = K

        return km

    def print_top_terms(self) -> None:
        """Prints the 10 top terms per cluster.
        """
        if self.svd:
            original_space_centroids = self.svd.inverse_transform(
                self.km.cluster_centers_)
            ordered_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            ordered_centroids = self.km.cluster_centers_.argsort()[:, ::-1]

        terms = self.vectorizer.get_feature_names_out()

        for k in range(self.K):
            print(f"Cluster {k}:")
            for i in ordered_centroids[k, :10]:
                print(" %s" % terms[i], end="")
            print()
            print()

    def print_most_representative_samples(self, X: list, num_samples: int) -> None:
        """Prints the most representative samples per cluster.

        Args:
            X (list): input data list.
            num_samples (int): the number of most representative samples to return.
        """
        predictions = self.km.predict(X)
        distances = self.km.transform(X)

        for k in range(self.K):
            ind = self.most_representative_sample_indices(
                distances, predictions, k, num_samples)

            print(f"Cluster {k}: most representative sample indices: {ind}")
            for i in ind:
                print(self.filtered_data[i])
                print()
            print()

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

    @staticmethod
    def plot_elbow_curve(max_k: int, X: list) -> None:
        """Plots elbow curve for all clusters until max_k.

        Args:
            max_k (int): the maximum cluster size to test.
            X (list): input data list.
        """
        plot_x = []
        for k in range(2, max_k):
            km = KMeans(n_clusters=k, init="k-means++")
            km.fit(X)
            plot_x.append(km.inertia_)

        plt.plot(range(2, max_k), plot_x)
        plt.grid(True)
        plt.title('Elbow curve')
        plt.xlabel('K (cluster size)')
        plt.ylabel('Sum of squared distances to closest center')
        plt.show()

    @staticmethod
    def plot_silhouette_analysis(max_k: int, X: list) -> None:
        """Plots the results of the silhouette analysis.

        Args:
            max_k (int): the maximum cluster size to test.
            X (list): input data list.
        """
        sil_scores = []
        for k in range(2, max_k):
            km = KMeans(n_clusters=k, init="k-means++")
            km.fit(X)
            sil_score = metrics.silhouette_score(X, km.labels_)
            sil_scores.append(sil_score)

        plt.plot(range(2, max_k), sil_scores)
        plt.grid(True)
        plt.title('Silhouette analysis')
        plt.xlabel('K (cluster size)')
        plt.ylabel('Silhouette coefficient')
        plt.show()

    @staticmethod
    def plot_calinski_harabasz_analysis(max_k: int, X: list) -> None:
        """Plots the results of the Calinski-Harabasz analysis.

        Args:
            max_k (int): the maximum cluster size to test.
            X (list): input data list.
        """
        sil_scores = []
        for k in range(2, max_k):
            km = KMeans(n_clusters=k, init="k-means++")
            km.fit(X)
            sil_score = metrics.calinski_harabasz_score(X, km.labels_)
            sil_scores.append(sil_score)

        plt.plot(range(2, max_k), sil_scores)
        plt.grid(True)
        plt.title('Calinski-Harabasz analysis')
        plt.xlabel('K (cluster size)')
        plt.ylabel('Calinski-Harabasz score')
        plt.show()

    @staticmethod
    def plot_davies_bouldin_score_analysis(max_k: int, X: list) -> None:
        """Plots the results of the Davies-Bouldin analysis.

        Args:
            max_k (int): the maximum cluster size to test.
            X (list): input data list.
        """
        sil_scores = []
        for k in range(2, max_k):
            km = KMeans(n_clusters=k, init="k-means++")
            km.fit(X)
            sil_score = metrics.calinski_harabasz_score(X, km.labels_)
            sil_scores.append(sil_score)

        plt.plot(range(2, max_k), sil_scores)
        plt.grid(True)
        plt.title('Davies-Bouldin analysis')
        plt.xlabel('K (cluster size)')
        plt.ylabel('Davies-Bouldin score')
        plt.show()
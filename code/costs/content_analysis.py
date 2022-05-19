from typing import List
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

# Implementation is based on:
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering

# Number of clusters
K = 4
# Filter on tweets that are hateful ('1') or not hateful ('0')
HS = '0'
# If HS is set to hateful, then we can filter on targeted to a specific individual ('1') or a generic group ('0')
TR = '0'
# If HS is set to hateful, then we can filter on aggressive tweets ('1') or non-aggressive tweets ('0')
AG = '0'
# Path of CSV file
FILE_PATH = "F:\Thesis\data\SemEval\hateval2019_en_train.csv"
# Number of most representative samples per cluster
NUM_SAMPLES = 5


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


with open(FILE_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

print("Original data length: ", len(data) - 1)

# Filter out
filtered_data = list(
    filter(lambda x: x[2] == HS and x[3] == TR and x[4] == AG, data))

# Remove first row since these contains headers
filtered_data = filtered_data[1:]

print("After applying filters: ", len(filtered_data))

# Remove all tweets that are invalid (contain urls, mentions, or not enough text after cleaning)
filtered_data = [
    x for x in filtered_data if valid_text(x[1])]

# Remove html attributes and clean tweets by removing hashtags, mentions, and urls
data = list(map(lambda x: p.clean(html.unescape(x[1])), filtered_data))

print("Data length after removing invalid tweets: ", len(data))

# Fit and transform the data using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data)

# Perform Latent Semantic Analysis (LSA) to reduce dimensions
# This helps to remove noise
# We use Singular Value Decomposition (SVD) on the TF-IDF data,
# which is then known as LSA.
# For LSA, a dimensionality of 100 is recommended.
svd = TruncatedSVD(n_components=100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

# plot_x = []

# for k in range(2, 100):
#     # Apply k-means clustering
#     km = KMeans(n_clusters=k, init="k-means++")

#     # Calculate distances between samples and all clusters
#     distances = km.fit_transform(X)
#     original_space_centroids = svd.inverse_transform(km.cluster_centers_)
#     ordered_centroids = original_space_centroids.argsort()[:, ::-1]
#     terms = vectorizer.get_feature_names_out()

#     plot_x.append(km.inertia_)

# plt.plot(range(2, 100), plot_x)
# plt.grid(True)
# plt.title('Elbow curve')
# plt.show()

km = KMeans(n_clusters=K, init="k-means++")

# Calculate distances between samples and all clusters
predictions = km.fit_predict(X)
distances = km.transform(X)
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
ordered_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for k in range(K):
    print("=======================================")
    print("Cluster ", k)
    # Print top 10 terms per cluster
    print("Top terms:")
    for i in ordered_centroids[k, :10]:
        print(" %s" % terms[i], end="")
    print()
    print()

    ind = most_representative_sample_indices(
        distances, predictions, k, NUM_SAMPLES)

    print("Most representative sample indices:", ind)
    for i in ind:
        print(data[i])
        print(filtered_data[i])
        print()
    print()

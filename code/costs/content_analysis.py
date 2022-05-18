from dataclasses import field
from pydoc import doc
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import preprocessor as p
import html
import numpy as np

# Implementation is based on:
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering

# Number of clusters
K = 2
# Filter on tweets that are hateful ('1') or not hateful ('0')
HS = '1'
# If HS is set to hateful, then we can filter on targeted to a specific individual ('1') or a generic group ('0')
TR = '1'
# If HS is set to hateful, then we can filter on aggressive tweets ('1') or non-aggressive tweets ('0')
AG = '1'
# Path of CSV file
FILE_PATH = "F:\Thesis\data\SemEval\hateval2019_en_train.csv"
# Number of most representative samples per cluster
NUM_SAMPLES = 5


def most_representative_sample_indices(distances, cluster_index, num_samples):
    distances = list(map(lambda s: s[cluster_index], distances))
    sorted_distances_ind = np.argsort(np.array(distances))
    return sorted_distances_ind[0:num_samples]


def contains_mention_or_url(text):
    tokenized_text = p.tokenize(text)
    cleaned_text = p.clean(html.unescape(text))
    return "$MENTION$" in tokenized_text or "$URL$" in tokenized_text or cleaned_text == ''


with open(FILE_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

# Filter out
filtered_data = list(
    filter(lambda x: x[2] == HS and x[3] == TR and x[4] == AG, data))
filtered_data = filtered_data[1:]
# print("size: ", len(filtered_data))

# data = data[:10]
# print("data: ", data)
# tokenized_tweets = list(map(lambda x: p.tokenize(x[1]), data))
# tokenized_tweets_with_index = [{'tweet': x, 'index': i}
#                                for i, x in enumerate(tokenized_tweets)]
filtered_data = [
    x for x in filtered_data if contains_mention_or_url(x[1]) == False]

# Remove html attributes and clean tweets by removing hashtags, mentions, and urls
# data_with_index = list(map(lambda x: {'tweet': p.clean(html.unescape(
#     x['tweet'])), 'index': x['index']}, filtered_tokenized_tweets))
data = list(map(lambda x: p.clean(html.unescape(x[1])), filtered_data))

# Fit and transform the data using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")
X = vectorizer.fit_transform(data)

# Apply k-means clustering
km = KMeans(
    n_clusters=K,
    init="k-means++",
)
pred = km.fit_predict(X)
dist = km.transform(X)
# print("predictions: ", pred)
# print("distances: ", dist)

# Print top 10 terms per cluster
print("Top terms per cluster:")
ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(K):
    print("Cluster %d:" % i, end="")
    for ind in ordered_centroids[i, :10]:
        print(" %s" % terms[ind], end="")
    print()

for k in range(K):
    ind = most_representative_sample_indices(dist, k, NUM_SAMPLES)
    print("=======================================")
    print("Cluster ", k)
    print("Most representative sample indices:", ind)
    for i in ind:
        print(data[i])
        print(filtered_data[i])
        print()
    print()

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import preprocessor as p
import html

# Implementation is based on:
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering

# Number of clusters
K = 8
# The nu be
N_COMPONENTS = 100
# Filter on tweets that are hateful ('1') or not hateful ('0')
HS = '0'
# If HS is set to hateful, then we can filter on targeted to a specific individual ('1') or a generic group ('0')
TR = '0'
# If HS is set to hateful, then we can filter on aggressive tweets ('1') or non-aggressive tweets ('0')
AG = '0'
# Path of CSV file
FILE_PATH = "F:\Thesis\data\SemEval\hateval2019_en_train.csv"

with open(FILE_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

# Filter out
filtered_data = list(
    filter(lambda x: x[2] == HS and x[3] == TR and x[4] == AG, data))
print("size: ", len(filtered_data))

# Remove html attributes and clean tweets by removing hashtags, mentions, and urls
data = list(map(lambda x: p.clean(html.unescape(x[1])), filtered_data[1:]))

# Fit and transform the data using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")
X = vectorizer.fit_transform(data)

# Apply k-means clustering
km = KMeans(
    n_clusters=K,
    init="k-means++",
)
pred = km.fit_predict(X)

# Print top 10 terms per cluster
print("Top terms per cluster:")
ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(K):
    print("Cluster %d:" % i, end="")
    for ind in ordered_centroids[i, :10]:
        print(" %s" % terms[ind], end="")
    print()

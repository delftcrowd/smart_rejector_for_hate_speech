from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import preprocessor as p
import html

# Implementation is based on:
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering

K = 10
N_COMPONENTS = 100
FILE_PATH = "F:\Thesis\data\SemEval\hateval2019_en_train.csv"

with open(FILE_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

filtered_data = list(filter(lambda x: x[2] == '0', data))
print("size: ", len(filtered_data))

data = list(map(lambda x: p.clean(html.unescape(x[1])), filtered_data[1:]))
print(data[:10])
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")

X = vectorizer.fit_transform(data)

svd = TruncatedSVD(n_components=N_COMPONENTS)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

km = KMeans(
    n_clusters=K,
    init="k-means++",
    max_iter=100,
    n_init=1,
)
pred = km.fit_predict(X)

print("Top terms per cluster:")
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(K):
    print("Cluster %d:" % i, end="")
    for ind in order_centroids[i, :10]:
        print(" %s" % terms[ind], end="")
    print()

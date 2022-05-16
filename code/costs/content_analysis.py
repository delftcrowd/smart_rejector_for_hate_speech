from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

K = 10
N_COMPONENTS = 100

dataset = fetch_20newsgroups(subset="all", shuffle=True, random_state=42)

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")

X = vectorizer.fit_transform(dataset.data[:1000])

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

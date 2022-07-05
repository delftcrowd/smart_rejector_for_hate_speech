import unittest
from kmeans import KMeansClustering
from sklearn.datasets import fetch_20newsgroups
import numpy as np


class TestKMeansClustering(unittest.TestCase):
    def setUp(self):
        categories = [
            "alt.atheism",
            "talk.religion.misc",
            "comp.graphics",
            "sci.space",
        ]

        self.dataset = fetch_20newsgroups(
            subset="all", categories=categories, shuffle=True, random_state=42
        )
        self.data = self.dataset.data
        self.labels = self.dataset.target
        self.true_k = np.unique(self.labels).shape[0]

    def test_cluster(self):
        km = KMeansClustering()
        X = km.fit_tfidf(self.data)
        X = km.fit_lsa(X)
        km.cluster(X=X, K=self.true_k)

        text1 = "atheism"
        text2 = "atheists"
        text3 = "space"
        transformed_texts = km.vectorizer.transform([text1, text2, text3])
        transformed_texts = km.lsa.transform(transformed_texts)
        [pred1, pred2, pred3] = km.km.predict(transformed_texts)
        self.assertEqual(pred1, pred2)
        self.assertNotEqual(pred1, pred3)

    def test_most_representative_sample_indices_1(self):
        distances = [[2, 0.9], [0.8, 2], [2, 0.7], [0.4, 2], [2, 0.2], [0.1, 2]]
        predictions = [1, 0, 1, 0, 1, 0]
        cluster_index = 0
        num_samples = 2
        indices = KMeansClustering.most_representative_sample_indices(
            distances, predictions, cluster_index, num_samples
        )
        self.assertListEqual(list(indices), [5, 3])

    def test_most_representative_sample_indices_2(self):
        distances = [[2, 0.9], [0.8, 2], [2, 0.7], [0.4, 2], [2, 0.2], [0.1, 2]]
        predictions = [1, 0, 1, 0, 1, 0]
        cluster_index = 1
        num_samples = 2
        indices = KMeansClustering.most_representative_sample_indices(
            distances, predictions, cluster_index, num_samples
        )
        self.assertListEqual(list(indices), [4, 2])

    def test_most_representative_sample_indices_3(self):
        distances = [[2, 0.9], [0.8, 2], [2, 0.7], [0.4, 2], [2, 0.2], [0.1, 2]]
        predictions = [1, 0, 1, 0, 1, 0]
        cluster_index = 1
        num_samples = 3
        indices = KMeansClustering.most_representative_sample_indices(
            distances, predictions, cluster_index, num_samples
        )
        self.assertListEqual(list(indices), [4, 2, 0])


if __name__ == "__main__":
    unittest.main()

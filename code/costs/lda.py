from re import S
from sklearn.decomposition import LatentDirichletAllocation
from typing import Any, List
from sklearn.feature_extraction.text import CountVectorizer


class LDA:
    """Latent Dirichlet Allocation (LDA) class.
    """

    def __init__(self) -> None:
        """Initializes the LDA object.
        """        
        self.lda = None
        self.vectorizer = None

    def fit_tf(self, X: List[str]) -> Any:
        """Fits the CountVectorizer on the data.

        Args:
            X (List[str]): input data.

        Returns:
            Any: matrix of token counts
        """        
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(X)
        self.vectorizer = vectorizer
        return X

    def fit_lda(self, X: List[str], n_components: int = 10) -> Any:
        """Fits the LDA method on the data.

        Args:
            X (List[str]): input data list.
            n_components (int, optional): number of topics. Defaults to 10.

        Returns:
            Any: the fitted LDA object.
        """
        lda = LatentDirichletAllocation(n_components=n_components)
        lda.fit(X)
        self.lda = lda
        return lda

    def print_top_terms(self) -> None:
        """Prints the 10 top words per topic.
        """
        feature_names = self.vectorizer.get_feature_names_out()

        for index, topic in enumerate(self.lda.components_):
            top_features_ind = topic.argsort()[: -10 - 1: -1]
            top_features = [feature_names[i] for i in top_features_ind]
            print("Topic: ", index)
            print(top_features)

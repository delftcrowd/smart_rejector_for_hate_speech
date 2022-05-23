from gensim.models import HdpModel
from typing import List
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_documents


class HDP:
    """
    """

    def __init__(self, X: List[str]) -> None:
        """Initializes 
        """
        # Implementation from: https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html

        processed_corpus = preprocess_documents(X)
        print(processed_corpus[0])
        self.corpus = processed_corpus
        self.dictionary = corpora.Dictionary(processed_corpus)
        self.bow_corpus = [self.dictionary.doc2bow(
            text) for text in self.corpus]
        self.model = HdpModel(self.bow_corpus, self.dictionary)

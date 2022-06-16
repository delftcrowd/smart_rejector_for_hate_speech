from transformers import AutoTokenizer, TFAutoModel
from os import environ as env
env['CUDA_VISIBLE_DEVICES'] = '0,1'


class DistilBERT:
    """DistilBERT model.
    """

    def __init__(self):
        """Initializes the DistilBERT model.
        """

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = TFAutoModel.from_pretrained("bert-base-uncased")

    def fit(self, X: list, y: list):
        inputs = self.tokenizer("Hello world!", return_tensors="tf")

    def predict(self, X: list) -> list:
        """Creates a list of predictions for a list of data samples.

        Args:
            X (list): list of data samples.

        Returns:
            list: prediction classes.
        """
        predictions = self.cnn.predict(X)
        return np.argmax(predictions, axis=1)

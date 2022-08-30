import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.engine.input_layer import Input
from keras.layers import (
    Conv1D,
    GlobalMaxPooling1D,
    Concatenate,
    Embedding,
    Dense,
    Dropout,
    TextVectorization,
)
from keras.regularizers import l2
from keras.models import Model
import keras
import keras_tuner as kt


class HyperCNN(kt.HyperModel):
    """Class for performing hyperparameter tuning on the CNN model."""

    def __init__(
        self,
        num_classes: int,
        vocab_len: int,
        embed_size: int,
        text_vectorizer,
        loss_type: str,
    ):
        self.num_classes = num_classes
        self.vocab_len = vocab_len
        self.embed_size = embed_size
        self.text_vectorizer = text_vectorizer
        self.loss_type = loss_type

    def build(self, hp):
        """Returns the CNN model from the work of Agrawal and Awekar for hyperparameter tuning.
        https://github.com/sweta20/Detecting-Cyberbullying-Across-SMPs/blob/master/models.py

        Returns:
            Model: the CNN model builder.
        """
        learning_rate = hp.Choice("learning_rate", values=[0.1, 0.001, 0.0001, 0.00001])

        _input = Input(shape=(1,), dtype="string")
        pre = self.text_vectorizer(_input)

        emb = Embedding(
            self.text_vectorizer.vocabulary_size(), self.embed_size, trainable=True
        )(pre)
        x = Dropout(0.25)(emb)
        x1 = Conv1D(
            self.embed_size,
            3,
            padding="valid",
            kernel_regularizer=l2(0.01),
            activation="relu",
        )(x)
        x2 = Conv1D(
            self.embed_size,
            4,
            padding="valid",
            kernel_regularizer=l2(0.01),
            activation="relu",
        )(x)
        x3 = Conv1D(
            self.embed_size,
            5,
            padding="valid",
            kernel_regularizer=l2(0.01),
            activation="relu",
        )(x)
        x = Concatenate(axis=1)([x1, x2, x3])
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if self.loss_type == "logits":
            custom_loss = CategoricalCrossentropy(from_logits=True)
            model = Dense(self.num_classes)(x)
            model = Model(_input, model)
            model.compile(loss=custom_loss, optimizer=adam, metrics=["accuracy"])
        if self.loss_type == "softmax":
            model = Dense(self.num_classes, activation="softmax")(x)
            model = Model(_input, model)
            model.compile(
                loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
            )

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32, 64, 128]),
            epochs=hp.Choice("epochs", [2, 5, 10, 20]),
            **kwargs,
        )

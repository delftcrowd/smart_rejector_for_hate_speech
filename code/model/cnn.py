import tensorflow as tf
import numpy as np
from keras.losses import CategoricalCrossentropy
from keras.engine.input_layer import Input
from keras.layers import Conv1D, GlobalMaxPooling1D, Concatenate, Embedding, Dense, Dropout, TextVectorization
from keras.regularizers import l2
from keras.models import Model
import keras


class CNN:
    """Convolutional Neural Network.
    """

    def __init__(self,
                 max_len: int,
                 num_classes: int,
                 vocab_len: int,
                 batch_size: int,
                 epochs: int,
                 embed_size: int,
                 loss_type: str = "softmax",
                 checkpoint_path: str = "results/cp.ckpt",
                 save_path: str = "results/model.tf",
                 save_model: bool = False,
                 text_vectorizer: any = None,
                 embedding_matrix: any = None
                 ):
        """Initializes the CNN model.

        Args:
            max_len (int): the maximum length of a data sample.
            num_classes (int): the number of classes.
            vocab_len (int): the size of the vocabulary.
            batch_size (int): the batch size.
            epochs (int): the number of epochs.
            embed_size (int): the embed size.
            checkpoint_path (str, optional): the path to save each checkpoint. Defaults to "results/cp.ckpt".
            save_path (str, optional): the path to save the final model. Defaults to "results/model.h5".
            loss_type (str, optional): either 'logits' or 'softmax'. Defaults to 'softmax'.
            save_model (boolean, optional): whether the model needs to be saved or not.
            text_vectorizer (any, optional): the TextVectorization layer.
            embedding_matrix (any, optional): pretrained embedding matrix.
        """

        self.max_len = max_len
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.embed_size = embed_size
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.vocab_len = vocab_len
        self.loss_type = loss_type
        self.save_model = save_model
        self.text_vectorizer = text_vectorizer
        self.cnn = None
        self.embedding_matrix = embedding_matrix

    def fit(self, X: list, y: list) -> Model:
        """Fits the CNN model with the list of data samples X
        and its labels y.

        Args:
            X (list): list of data samples.
            y (list): list of labels.

        Returns:
            Model: keras Model.
        """
        if self.text_vectorizer is None:
            self.text_vectorizer = TextVectorization(output_mode="int")
            self.text_vectorizer.adapt(X)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        if self.save_model:
            callbacks = [cp_callback]
        else:
            callbacks = []

        validation_data = None

        model = self.model()
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                  shuffle=True, verbose=1, callbacks=callbacks,
                  validation_data=validation_data)

        if self.save_model:
            model.save(self.save_path)

        self.cnn = model
        return model

    def load_weights(self, checkpoint_path: str) -> Model:
        """Loads weights from the checkpoint path into the model.

        Args:
            checkpoint_path (str): the checkpoint path.

        Returns:
            Model: the loaded model.
        """
        model = self.model()
        model.load_weights(checkpoint_path)
        self.cnn = model
        return model

    @staticmethod
    def load(save_path: str) -> Model:
        """Loads a model from the save path.

        Args:
            save_path (str): the save path.

        Returns:
            Model: the loaded model.
        """
        model = tf.keras.models.load_model(save_path, compile=False)
        return model

    def predict(self, X: list) -> list:
        """Creates a list of predictions for a list of data samples.

        Args:
            X (list): list of data samples.

        Returns:
            list: prediction classes.
        """
        predictions = self.cnn.predict(X)
        return np.argmax(predictions, axis=1)

    def model(self) -> Model:
        """Returns the CNN model from the work of Agrawal and Awekar.
        https://github.com/sweta20/Detecting-Cyberbullying-Across-SMPs/blob/master/models.py

        Returns:
            Model: the CNN model.
        """
        _input = Input(shape=(1,), dtype="string")
        pre = self.text_vectorizer(_input)

        if self.embedding_matrix is not None:
            emb = Embedding(
                self.vocab_len,
                self.embed_size,
                embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
                trainable=False,
            )(pre)
        else:
            emb = Embedding(self.text_vectorizer.vocabulary_size(),
                            self.embed_size, trainable=True)(pre)
        x = Dropout(0.25)(emb)
        x1 = Conv1D(self.embed_size, 3, padding='valid', kernel_regularizer=l2(.01),
                    activation='relu')(x)
        x2 = Conv1D(self.embed_size, 4, padding='valid', kernel_regularizer=l2(.01),
                    activation='relu')(x)
        x3 = Conv1D(self.embed_size, 5, padding='valid', kernel_regularizer=l2(.01),
                    activation='relu')(x)
        x = Concatenate(axis=1)([x1, x2, x3])
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        if self.loss_type == "logits":
            custom_loss = CategoricalCrossentropy(from_logits=True)
            model = Dense(self.num_classes)(x)
            model = Model(_input, model)
            model.compile(loss=custom_loss,
                          optimizer='adam',
                          metrics=['accuracy'])
        if self.loss_type == "softmax":
            model = Dense(self.num_classes, activation='softmax')(x)
            model = Model(_input, model)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

        return model

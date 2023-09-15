import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.metrics import classification_report

class SpamDetectorModel:
    """
    A class to build, train, and evaluate an LSTM model for spam detection.
    """

    def __init__(self, max_vocab_length, output_dim):
        self.max_vocab_length = max_vocab_length
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        """
        Build and compile the LSTM model.
        """
        text_vectorization = TextVectorization(max_tokens=self.max_vocab_length,
                                               standardize="lower_and_strip_punctuation",
                                               split="whitespace",
                                               ngrams=None,
                                               output_mode='int',
                                               output_sequence_length=None)  # We'll set it later
        
        embedding = layers.Embedding(input_dim=self.max_vocab_length,
                                     output_dim=self.output_dim,
                                     embeddings_initializer="uniform",
                                     input_length=None)  # We'll set it later

        inputs = layers.Input(shape=(1,), dtype='string')
        x = text_vectorization(inputs)
        x = embedding(x)
        x = layers.LSTM(64)(x)
        outputs = layers.Dense(3, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs, name="LSTM_model")
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=5):
        """
        Train the model.
        """
        self.model.fit(X_train,
                       y_train,
                       epochs=epochs,
                       validation_data=(X_val, y_val))

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and print a classification report.
        """
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred_classes))

    def save(self, filepath):
        """
        Save the model to the given filepath.
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model from the given filepath.
        """
        loaded_model = load_model(filepath)
        instance = cls()
        instance.model = loaded_model
        return instance
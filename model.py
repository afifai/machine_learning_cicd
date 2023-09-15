import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

class SpamDetectorModel:
    def __init__(self, max_vocab_length=None, output_dim=None):
        if max_vocab_length is not None and output_dim is not None:
            self._build_model(max_vocab_length, output_dim)
    
    def _build_model(self, max_vocab_length, output_dim):
        """
        Build and compile the Keras model.
        """
        self.text_vectorization = TextVectorization(max_tokens=max_vocab_length,
                                                    standardize="lower_and_strip_punctuation",
                                                    split="whitespace",
                                                    output_mode='int')
        
        self.embedding = layers.Embedding(input_dim=max_vocab_length,
                                          output_dim=output_dim,
                                          embeddings_initializer="uniform")
        
        self.embedding.trainable = False
        
        inputs = layers.Input(shape=(1,), dtype='string')
        x = self.text_vectorization(inputs)
        x = self.embedding(x)
        x = layers.LSTM(64)(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs, name="LSTM_model")
        
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=["accuracy"])
        
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model on the provided data.
        """
        self.text_vectorization.adapt(X_train)
        self.model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    def evaluate_model(self, X_test, y_test, output_file=None):
        """
        Evaluate the model and print a classification report.
        Optionally, save the report to a text file.
        """
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        report_str = "Classification Report:\n"
        report_str += classification_report(y_true, y_pred_classes)
        
        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(report_str)
        
        print(report_str)
    
    def save(self, filepath):
        """
        Save the model to the given filepath.
        """
        self.model.save(filepath, save_format='tf')
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model from the given filepath.
        """
        instance = cls()
        instance.model = load_model(filepath, custom_objects={'TextVectorization': TextVectorization})
        return instance

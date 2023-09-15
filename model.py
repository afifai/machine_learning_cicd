import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

class SpamDetectorModel:
    def __init__(self, max_vocab_length=None, output_dim=None):
        if max_vocab_length is not None and output_dim is not None:
            self._build_model(max_vocab_length, output_dim)
    
    def _build_model(self, max_vocab_length, output_dim):
        self.text_vectorization = layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_vocab_length,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            output_mode='int'
        )
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
                           optimizer='adam',
                           metrics=["accuracy"])

    def train(self, X_train, y_train, X_val, y_val):
        self.text_vectorization.adapt(X_train)
        history = self.model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

        # Plot training & validation accuracy values
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig('outputs/accuracy_plot.png')

        # Plot training & validation loss values
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig('outputs/loss_plot.png')
        
        # Save model architecture as png
        plot_model(self.model, to_file='outputs/model_architecture.png', show_shapes=True, show_layer_names=True)

    def evaluate_model(self, X_test, y_test, branch='main'):
        """
        Evaluate the model and print a classification report.
        Optionally, save the report to a text file.
        """

        # Predict and extract the classes
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Generate and format the classification report
        report_dict = classification_report(y_true, y_pred_classes, output_dict=True)
        report_str = self._format_classification_report(report_dict)

        # Save report to text and pickle files
        self._save_report_to_file(report_str, "outputs/evaluation_report.txt")
        self._save_metrics_to_pickle(report_dict, f"outputs/metrics_{branch}.pkl")

        # Print the formatted report
        print(report_str)

    def _format_classification_report(self, report_dict):
        return f"""Classification Report:
    | Metrics     | Main Branch   |
    |-------------|---------------|
    | Accuracy    | {report_dict['accuracy']:.2f}          |
    | Precision   | {report_dict['1']['precision']:.2f}    |
    | Recall      | {report_dict['1']['recall']:.2f}       |
    | F1-Score    | {report_dict['1']['f1-score']:.2f}     |

    """

    def _save_report_to_file(self, report_str, filepath):
        with open(filepath, "w") as f:
            f.write(report_str)

    def _save_metrics_to_pickle(self, report_dict, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(report_dict, f)
    
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

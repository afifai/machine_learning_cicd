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
        x = layers.GRU(64)(x)
        outputs = layers.Dense(3, activation='softmax')(x)

        self.model = tf.keras.Model(inputs, outputs, name="LSTM_model")
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=["accuracy"])

    def train(self, X_train, y_train, X_val, y_val, branch='main'):
        self.text_vectorization.adapt(X_train)
        history = self.model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

        # Plot training & validation accuracy values
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model Accuracy {branch.capitalize()}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(f'outputs/accuracy_plot_{branch}.png')

        # Plot training & validation loss values
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss {branch.capitalize()}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(f'outputs/loss_plot_{branch}.png')
        
        # Save model architecture as png
        plot_model(self.model, to_file=f'outputs/model_architecture_{branch}.png', show_shapes=True, show_layer_names=True)

    def evaluate_model(self, X_test, y_test, branch='main', stage='validation'):
        """
        Evaluate the model and print a classification report.
        Optionally, save the report to a text file.
        """

        # Predict and extract the classes
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        if stage == 'validation':
            y_test = np.argmax(y_test, axis=1)

        # Generate and format the classification report
        report_dict = classification_report(y_test, y_pred_classes, output_dict=True)
        report_str = self._format_classification_report(report_dict, stage, branch)

        # Save report to text and pickle files
        self._save_report_to_file(report_str, f"outputs/evaluation_report_{branch}_{stage}.txt")
        self._save_metrics_to_pickle(report_dict, f"outputs/metrics_{branch}_{stage}.pkl")

        # Print the formatted report
        print(report_str)

    def _format_classification_report(self, report_dict, stage, branch, report_main=None):
        if branch == 'main':
            report = [f"Classification Report {stage.capitalize()}:",
                    "| Metrics     | Main Branch   |",
                    "|-------------|---------------|",
                    f"| Accuracy    | {report_dict['accuracy']:.2f}          |",
                    f"| Precision   | {report_dict['1']['precision']:.2f}    |",
                    f"| Recall      | {report_dict['1']['recall']:.2f}       |",
                    f"| F1-Score    | {report_dict['1']['f1-score']:.2f}     |",
                    "\n",
            ]
        elif branch == 'experiment':
            report_main = self._load_report(f"outputs/metrics_main_{stage}.pkl")
            report = [f"Classification Report {stage.capitalize()}:",
                    "| Metrics     | Experiment Branch   | Main Branch   |",
                    "|-------------|---------------|---------------|",
                    f"| Accuracy    | {report_dict['accuracy']:.2f}          | {report_main['accuracy']:.2f}          |",
                    f"| Precision   | {report_dict['1']['precision']:.2f}    | {report_main['1']['precision']:.2f}    |",
                    f"| Recall      | {report_dict['1']['recall']:.2f}       | {report_main['1']['recall']:.2f}       |",
                    f"| F1-Score    | {report_dict['1']['f1-score']:.2f}     | {report_main['1']['f1-score']:.2f}     |",
                    "\n",
            ]
        return '\n'.join(report)

    def _save_report_to_file(self, report_str, filepath):
        with open(filepath, "w") as f:
            f.write(report_str)

    def _save_metrics_to_pickle(self, report_dict, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(report_dict, f)
    
    def _load_report(self, filepath):
        with open(filepath, "rb") as f:
            report = pickle.load(f)
        return report
    
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
        instance.model = load_model(filepath)
        return instance

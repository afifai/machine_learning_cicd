import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_dataset(path='data/train.csv'):
    """
    Load dataset from a given path.
    """
    return pd.read_csv(path)

def load_test_data(path='data/test.csv'):
    """
    Load dataset from a given path.
    """
    df = pd.read_csv(path)
    X, y = df.Teks, df.label
    return X, y

def prepare_target(y_train, y_test):
    """
    One-hot encode target variables.
    """
    y_train_enc = to_categorical(y_train)
    y_test_enc = to_categorical(y_test)
    return y_train_enc, y_test_enc

def split_data(df):
    """
    Split the dataset into training and test sets.
    """
    train_df, test_df = train_test_split(df, test_size=0.2)
    X_train, y_train = train_df.Teks, train_df.label
    X_test, y_test = test_df.Teks, test_df.label
    y_train, y_test = prepare_target(y_train, y_test)
    return X_train, X_test, y_train, y_test


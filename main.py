import argparse
from utils.data_utils import load_dataset, split_data, load_test_data
from model import SpamDetectorModel

def train():
    # Load and prepare data
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Initialize and train the model
    model = SpamDetectorModel(max_vocab_length=4000, output_dim=128)
    
    # Adapt TextVectorization layer
    model.model.layers[1].adapt(X_train)
    
    # Train the model
    model.train(X_train, y_train, X_test, y_test)
    
    # Save the model
    model.save("models/model.tf")
    
    # Evaluate the model
    model.evaluate_model(X_test, y_test)

def test():
    # Load data testing
    X_test, y_test = load_test_data()
    model = SpamDetectorModel.load('model.tf')
    # Evaluate the model
    model.evaluate_model(X_test, y_test, stage='test')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--test', action='store_true')
    args = vars(ap.parse_args())

    if args['train']:
        train()
    
    elif args['test']:
        test()

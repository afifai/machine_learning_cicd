import argparse
from utils.data_utils import load_dataset, split_data, load_test_data
from model import SpamDetectorModel

def train(branch='main'):
    # Load and prepare data
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Initialize and train the model
    model = SpamDetectorModel(max_vocab_length=4000, output_dim=128)
    
    # Adapt TextVectorization layer
    model.model.layers[1].adapt(X_train)
    
    # Train the model
    model.train(X_train, y_train, X_test, y_test, branch)
    
    # Save the model
    model.save(f"models/model_{branch}.tf")
    
    # Evaluate the model
    model.evaluate_model(X_test, y_test, branch)

def test(branch='main'):
    # Load data testing
    X_test, y_test = load_test_data()
    model = SpamDetectorModel.load(f'model_{branch}.tf')
    # Evaluate the model
    model.evaluate_model(X_test, y_test, branch, stage='test')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--experiment', action='store_true')
    args = vars(ap.parse_args())

    branch = 'experiment' if args['experiment'] else 'main'

    if args['train']:
        train(branch)
    
    elif args['test']:
        test(branch)

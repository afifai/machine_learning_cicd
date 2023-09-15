from utils.data_utils import load_dataset, split_data
from model import SpamDetectorModel

def main():
    # Load and prepare data
    url = "https://raw.githubusercontent.com/afifai/hacktiv8-trial-class-spamdetector/main/data/dataset_sms_spam%20_v1.csv"
    df = load_dataset(url)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Initialize and train the model
    model = SpamDetectorModel(max_vocab_length=4000, output_dim=128)
    
    # Adapt TextVectorization layer
    model.model.layers[1].adapt(X_train)
    
    # Train the model
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate the model
    model.evaluate_model(X_test, y_test)

if __name__ == "__main__":
    main()

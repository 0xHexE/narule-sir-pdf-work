#!/usr/bin/env python3
"""
Federated Learning Client CLI for connecting to Flower server.
Supports loading data from CSV files and connecting to a server.
"""

import argparse
import logging
import sys

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_process_data(csv_file_path: str):
    """Load and preprocess the CSV data for federated learning."""
    try:
        logger.info(f"Loading data from {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        logger.debug(f"Raw data shape: {df.shape}")
        logger.debug(f"Data columns: {list(df.columns)}")
        
        # Dropping features with data outside 98% confidence interval
        logger.info("Filtering data outside 98% confidence interval")
        df1 = df.copy()
        logger.debug(f"Original data points: {len(df)}")
        
        for feature in df1.columns[:-2]:
            lower_range = np.quantile(df[feature], 0.01)
            upper_range = np.quantile(df[feature], 0.99)
            logger.debug(f"Feature {feature}: range [{lower_range:.4f}, {upper_range:.4f}]")
            dropped_count = len(df1[(df1[feature] > upper_range) | (df1[feature] < lower_range)])
            if dropped_count > 0:
                logger.debug(f"Dropping {dropped_count} outliers for feature {feature}")
            df1 = df1.drop(df1[(df1[feature] > upper_range) | (df1[feature] < lower_range)].index, axis=0)
        
        logger.debug(f"Data points after filtering: {len(df1)}")
        
        # Split data into train and test
        train = df1.sample(frac=0.8, random_state=200)
        test = df1.drop(train.index)
        logger.debug(f"Training set size: {len(train)}")
        logger.debug(f"Test set size: {len(test)}")
        
        # Standardize the data
        logger.info("Standardizing data")
        allmeans = []
        allstds = []
        
        def get_data(data):
            X_train = data.drop(['Activity', 'subject'], axis=1)
            y_train = data['Activity']
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            allmeans.append(scaler.mean_)
            allstds.append(scaler.scale_)
            return X_train, y_train
        
        X_train, y_train = get_data(df1)
        meant = np.mean(allmeans, axis=0)
        stdt = np.mean(allstds, axis=0)
        logger.debug(f"Mean scaling values: {meant}")
        logger.debug(f"Std scaling values: {stdt}")
        
        X_test = test.drop(['Activity', 'subject'], axis=1)
        X_test1 = np.array(X_test)
        X_test1 = (X_test1 - meant) / stdt
        X_test = pd.DataFrame(X_test1, columns=X_test.columns)
        y_test = test['Activity']
        logger.debug(f"Test features shape: {X_test.shape}")
        logger.debug(f"Test labels shape: {y_test.shape}")
        
        # Create time series dataset for sequence modeling
        logger.info("Creating time series dataset")
        def create_dataset(X, y, time_steps, step=1):
            Xs, ys = [], []
            for i in range(0, len(X) - time_steps, step):
                x = X.iloc[i:(i + time_steps)].values
                labels = y.iloc[i: i + time_steps]
                Xs.append(x)
                ys.append(stats.mode(labels)[0])
            return np.array(Xs), np.array(ys).reshape(-1, 1)
        
        X_train, y_train = create_dataset(X_train, y_train, 100, step=50)
        X_test, y_test = create_dataset(X_test, y_test, 100, step=50)
        
        logger.debug(f"Final training sequences: {X_train.shape}")
        logger.debug(f"Final training labels: {y_train.shape}")
        logger.debug(f"Final test sequences: {X_test.shape}")
        logger.debug(f"Final test labels: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading and processing data: {e}")
        raise

def create_model():
    """Create the neural network model architecture."""
    logger.info("Creating model architecture")
    model = keras.Sequential()
    model.add(layers.Input(shape=[100, 12]))
    logger.debug("Added input layer with shape [100, 12]")
    model.add(layers.Conv1D(filters=32, kernel_size=3, padding="same"))
    logger.debug("Added Conv1D layer with 32 filters")
    model.add(layers.BatchNormalization())
    logger.debug("Added BatchNormalization")
    model.add(layers.ReLU())
    logger.debug("Added ReLU activation")
    model.add(layers.Conv1D(filters=64, kernel_size=3, padding="same"))
    logger.debug("Added Conv1D layer with 64 filters")
    model.add(layers.BatchNormalization())
    logger.debug("Added BatchNormalization")
    model.add(layers.ReLU())
    logger.debug("Added ReLU activation")
    model.add(layers.MaxPool1D(2))
    logger.debug("Added MaxPool1D with pool size 2")
    model.add(layers.LSTM(64))
    logger.debug("Added LSTM layer with 64 units")
    model.add(layers.Dense(units=128, activation='relu'))
    logger.debug("Added Dense layer with 128 units")
    model.add(layers.Dense(13, activation='softmax'))
    logger.debug("Added output layer with 13 units and softmax activation")
    
    model.compile(
        optimizer="sgd",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    logger.debug("Model compiled with SGD optimizer and sparse categorical crossentropy loss")
    
    return model

class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning."""
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        logger.debug(f"Starting fit with config: {config}")
        logger.debug(f"Received parameters shape: {[p.shape for p in parameters]}")
        self.model.set_weights(parameters)
        logger.debug("Model weights set successfully")
        
        r = self.model.fit(
            self.X_train, self.y_train,
            epochs=5,
            validation_data=(self.X_test, self.y_test),
            verbose=1 if logger.level == logging.DEBUG else 0
        )
        hist = r.history
        logger.info(f"Fit history: {hist}")
        logger.debug(f"Training loss: {hist.get('loss', [])}")
        logger.debug(f"Training accuracy: {hist.get('sparse_categorical_accuracy', [])}")
        logger.debug(f"Validation loss: {hist.get('val_loss', [])}")
        logger.debug(f"Validation accuracy: {hist.get('val_sparse_categorical_accuracy', [])}")
        
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        logger.debug(f"Starting evaluation with config: {config}")
        logger.debug(f"Received parameters shape: {[p.shape for p in parameters]}")
        self.model.set_weights(parameters)
        logger.debug("Model weights set successfully")
        
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test,
                                           verbose=1 if logger.level == logging.DEBUG else 0)
        logger.info(f"Evaluation accuracy: {accuracy}")
        logger.debug(f"Evaluation loss: {loss}")
        
        return loss, len(self.X_test), {"accuracy": accuracy}

def main():
    """Main function to run the client."""
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--server-url', type=str, required=True,
                       help='Server URL to connect to (e.g., 127.0.0.1:8080)')
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--grpc-max-message-length', type=int, default=1024*1024*1024,
                       help='Maximum gRPC message length')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Set TensorFlow random seed for reproducibility
        tf.random.set_seed(42)
        
        # Load and process data
        X_train, y_train, X_test, y_test = load_and_process_data(args.data_file)
        
        # Create model
        model = create_model()
        
        # Create Flower client
        client = FlowerClient(model, X_train, y_train, X_test, y_test)
        
        logger.info(f"Connecting to server at {args.server_url}")
        logger.info(f"Using data file: {args.data_file}")
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.debug(f"Model summary: {model.summary()}")
        
        # Start Flower client
        fl.client.start_numpy_client(
            server_address=args.server_url,
            client=client,
            grpc_max_message_length=args.grpc_max_message_length
        )
        
    except Exception as e:
        logger.error(f"Client failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

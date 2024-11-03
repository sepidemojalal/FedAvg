import flwr as fl
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model.lstm_model import LSTMModel  # Import the LSTM model
from src.client import FLClient  # Import the FL client

def load_data(train_path, test_path):
    """Load and preprocess dataset."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data[['v1', 'delta_x', 'delta_v', 'a1']].values
    y_train = train_data['a1'].shift(-10).fillna(0).values

    X_test = test_data[['v1', 'delta_x', 'delta_v', 'a1']].values
    y_test = test_data['a1'].shift(-10).fillna(0).values

    return X_train, y_train, X_test, y_test

def start_federated_learning(X_train, y_train, X_test, y_test):
    """Start the Federated Learning process."""
    print("Starting Flower server...")
    client_data_size = len(X_train) // 2
    client_1_data = TensorDataset(X_train[:client_data_size], y_train[:client_data_size])
    client_2_data = TensorDataset(X_train[client_data_size:], y_train[client_data_size:])

    model = LSTMModel()

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            min_fit_clients=2,
            min_available_clients=2,
        )
    )

    print("Connecting clients to the server...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(model, client_1_data, client_2_data))

if __name__ == "__main__":
    train_path = "/mnt/data/Train_Aggregated.csv"
    test_path = "/mnt/data/Test_Aggregated.csv"
    
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    print("Data loaded")
    start_federated_learning(X_train, y_train, X_test, y_test)
    print("Finish training")

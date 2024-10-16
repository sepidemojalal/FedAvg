# server.py
import flwr as fl
from model.lstm_model import LSTMModel  # Import the LSTM model

def start_federated_learning():
    """Start the Federated Learning server."""
    model = LSTMModel()

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:5000",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            min_fit_clients=2,
            min_available_clients=2,
        )
    )

if __name__ == "__main__":
    print("Starting Flower server...")
    start_federated_learning()

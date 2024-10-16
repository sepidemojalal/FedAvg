# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model.lstm_model import LSTMModel

class FLClient(fl.client.NumPyClient):
    """Federated Learning Client."""
    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        for epoch in range(1):  # Train for 1 epoch
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch.unsqueeze(1))  
                loss = self.criterion(output.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        test_loader = DataLoader(self.test_data, batch_size=32)
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = self.model(X_batch.unsqueeze(1))  
                loss += self.criterion(output.squeeze(), y_batch).item()
        return loss / len(self.test_data), len(self.test_data), {}

def start_client():
    # Loading training data
    train_data_df = pd.read_csv("/Users/mojala15/federated/X_Train_V_id_19.csv") 
    # Assuming that 'a1' is the target variable and we have multiple input features
    X_train = train_data_df[['v1', 'delta_x', 'delta_v', 'a1']].values
    y_train = train_data_df['a1'].shift(-10).fillna(0).values  # Adjust as needed

    # Converting to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Creating TensorDataset
    train_data = TensorDataset(X_train_tensor, y_train_tensor)

    # Initializing model
    model = LSTMModel()

    # Starting the Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient(model, train_data))

# if __name__ == "__main__":
#     start_client()

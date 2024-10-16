import torch.nn as nn

class LSTMModel(nn.Module):
    """LSTM Model definition."""
    def __init__(self, input_size=4, hidden_size=50, num_layers=1, output_size=10):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=2*82, hidden_size=64, batch_first=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.relu1(x)
        x, _ = self.lstm2(x)
        x = self.relu2(x)
        x = self.fc1(x[:, -1, :])  # Only use the last step
        x = torch.sigmoid(x)
        return x

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1_l = nn.LSTM(input_size=2*81, hidden_size=64, batch_first=True)
        self.relu1_l = nn.ReLU(inplace=False)
        self.lstm2_l = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.relu2_l = nn.ReLU(inplace=False)

        self.lstm1_e = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.relu1_e = nn.ReLU(inplace=False)
        self.lstm2_e = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.relu2_e = nn.ReLU(inplace=False)

        self.fc1 = nn.Linear(in_features=32, out_features=1)

    def forward(self, l, e):

        # Landmark forward pass
        l, _ = self.lstm1_l(l)
        l = self.relu1_l(l)
        l, _ = self.lstm2_l(l)
        l = self.relu2_l(l)

        # # EEG forward pass
        # e, _ = self.lstm1_e(e)
        # e = self.relu1_e(e)
        # e, _ = self.lstm2_e(e)
        # e = self.relu2_e(e)

        # Concat and classify
        # out = self.fc1(torch.cat((l[:, -1, :], e[:, -1, :]), dim=1))
        out = self.fc1(l[:, -1, :])
        out = 10*torch.sigmoid(out)

        return out

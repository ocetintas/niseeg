import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # self.lstm1_l = nn.LSTM(input_size=2*81, hidden_size=64, batch_first=True)
        # self.relu1_l = nn.ReLU(inplace=False)
        # self.lstm2_l = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        # self.relu2_l = nn.ReLU(inplace=False)
        #
        # self.lstm1_e = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        # self.relu1_e = nn.ReLU(inplace=False)
        # self.lstm2_e = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        # self.relu2_e = nn.ReLU(inplace=False)

        # self.fc1 = nn.Linear(in_features=64, out_features=1)

        self.lstm1 = nn.LSTM(input_size=2*81, hidden_size=64, batch_first=True)
        self.relu1 = nn.ReLU(inplace=False)

        self.fc1 = nn.Linear(in_features=64, out_features=16)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(in_features=16, out_features=1)

    def forward(self, l, e):

        # # Landmark forward pass
        # l, _ = self.lstm1_l(l)
        # l = self.relu1_l(l)
        # l, _ = self.lstm2_l(l)
        # l = self.relu2_l(l)
        #
        # # # EEG forward pass
        # e, _ = self.lstm1_e(e)
        # e = self.relu1_e(e)
        # e, _ = self.lstm2_e(e)
        # e = self.relu2_e(e)

        # # Concat and classify
        # out = self.fc1(torch.cat((l[:, -1, :], e[:, -1, :]), dim=1))
        # out = 10*torch.sigmoid(out)
        #
        # return out

        # x = torch.cat((l, e), dim=2)
        x = l
        x, _ = self.lstm1(x)
        x = self.relu1(x[:, -1, :])
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = 10*torch.sigmoid(x)
        return x

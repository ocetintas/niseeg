import torch
import torch.nn as nn

class EmotionLSTM(nn.Module):
  def __init__(self, num_class):
    super().__init__()
    self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True) ##input : seq_len * batch * input_size
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
    self.sigmoid = nn.Sigmoid()
    self.dense = nn.Linear(in_features=32, out_features=num_class-1)
  def forward(self,x):
    x, _ = self.lstm1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x, _ = self.lstm2(x)
    x = self.sigmoid(x)
    x = self.dense(x)
    x = self.sigmoid(x)
    return x
from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, dropout=0., bidirectional=False, out_size=1,
                 length_prediction=24):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.length_prediction = length_prediction
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)
        # 定义全连接层
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # lstm_out, _ = self.lstm(x)
        # print('lstm_out', lstm_out.shape, x.shape, len(x))
        # pred = self.linear(lstm_out.reshape((len(x), -gru_32)))
        pred = self.linear(lstm_out[:, : self.length_prediction])
        return pred  # torch.Size([2, 24, gru_32])


import numpy as np
if __name__ == '__main__':
    a = torch.from_numpy(np.random.uniform(-1., 1., size=(2, 240, 14))).float()
    m = LSTM()
    print(m(a).shape)

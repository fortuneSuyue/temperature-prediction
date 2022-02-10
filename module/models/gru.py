from torch import nn
import torch


class GRU(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, dropout=0., bidirectional=False, out_size=1,
                 length_prediction=24):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.length_prediction = length_prediction
        self.bidirectional = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义全连接层
        self.linear = nn.Linear(hidden_size*self.bidirectional, out_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h = torch.zeros(self.num_layers * self.bidirectional, x.size(0), self.hidden_size).requires_grad_()
        out, h = self.gru(x, h.detach())
        # print(out.shape, h.shape)
        pred = self.linear(out[:, -self.length_prediction:, :])
        return pred  # torch.Size([2, 24, 1])


import numpy as np

if __name__ == '__main__':
    a = torch.from_numpy(np.random.uniform(-1., 1., size=(2, 240, 14))).float()
    m = GRU(bidirectional=False, num_layers=1, dropout=0.2)
    print(m(a).shape)

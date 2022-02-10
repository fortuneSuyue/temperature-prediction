from torch import nn
import torch

from module.models.gru import GRU
from module.models.self_attention import SelfAttentionV1


class AttentionGRUV1(GRU):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, dropout=0., bidirectional=False, out_size=1,
                 length_prediction=24):
        super(AttentionGRUV1, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                             dropout=dropout, bidirectional=bidirectional, out_size=out_size,
                                             length_prediction=length_prediction)
        self.attention_v1 = SelfAttentionV1(input_size=hidden_size*self.bidirectional)
        self.linear = nn.Linear(hidden_size*self.bidirectional, length_prediction)

    def forward(self, x, need_attention_value=False):
        # Initialize hidden state with zeros
        h = torch.zeros(self.num_layers * self.bidirectional, x.size(0), self.hidden_size).requires_grad_()
        out, h = self.gru(x, h.detach())
        # print(out.shape, h.shape)
        out, att_value = self.attention_v1(out)
        pred = self.linear(out.squeeze(-1)).unsqueeze(-1)
        # print(pred.shape, att_value.shape)
        if need_attention_value:
            return pred, att_value
        else:
            return pred  # torch.Size([2, 24, 1])


import numpy as np

if __name__ == '__main__':
    a = torch.from_numpy(np.random.uniform(-1., 1., size=(2, 240, 14))).float()
    m = AttentionGRUV1(bidirectional=False, num_layers=1, dropout=0.)
    print(m(a).shape)

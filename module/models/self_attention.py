import numpy as np
import torch
from torch import nn


class SelfAttentionV1(nn.Module):
    def __init__(self, input_size, batch_first=True):
        """
        This is a self-attention module for RNN. Better suited to classified tasks.
        :param input_size:
        """
        super(SelfAttentionV1, self).__init__()
        self.batch_first = batch_first
        self.w = nn.Linear(in_features=input_size, out_features=1, bias=False)

    def forward(self, h):
        """
        batch first or not. Reference: https://blog.csdn.net/imsuhxz/article/details/83058316
        :param h:
        :return:
        """
        if not self.batch_first:
            h = h.permute(1, 0, 2)
        b = h.size(0)
        x = self.w(h.contiguous().view(-1, h.size(-1)))
        x = x.view(b, -1)
        alpha = torch.softmax(x, dim=1)
        alpha = alpha.unsqueeze(-1)
        x = torch.bmm(h.permute(0, 2, 1), alpha)
        x = torch.tanh(x)
        return x, alpha


class BahdanauAttention(nn.Module):
    """
    Pytorch Code Reference: https://www.jianshu.com/p/4dd65ec4654c
    Paper:Bahdanau, D., K. Cho and Y. Bengio, Neural Machine Translation by Jointly Learning to Align and Translate. 2014.
    """
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int):
        """
        Related RNN should be bidirectional and Batch_first=True.
        :param enc_hidden_dim:
        :param dec_hidden_dim:
        """
        super(BahdanauAttention, self).__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.fc = nn.Linear(enc_hidden_dim*2+dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(dec_hidden_dim))

    def forward(self, hidden, enc_output):
        # hidden = [batch size,dec_hidden_dim]
        # enc_output = [batch size, step, enc_hidden_dim * 2]
        batch_size = hidden.size(0)
        step = enc_output.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, step, 1)
        # hidden = [batch size,step, dec_hidden_dim]
        # enc_output = [batch size, step, enc_hidden_dim * 2]
        # Calculate matching values of them
        energy = torch.tanh(self.fc(torch.cat([hidden, enc_output], dim=2)))
        energy = energy.permute(0, 2, 1)  # [batch size, dec_hidden_dim, step]
        # Contains a weight-sharing mechanism
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, dec_hidden_dim]
        attn = torch.bmm(v, energy).squeeze(1)  # [batch size, step]
        return torch.softmax(attn, dim=1)


if __name__ == '__main__':
    a = torch.from_numpy(np.random.uniform(-1., 1., size=(2, 240, 32))).float()
    m = SelfAttentionV1(input_size=32)
    m(a)

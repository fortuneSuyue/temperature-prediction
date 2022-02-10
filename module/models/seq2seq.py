import random

import torch
from torch import nn


class Encoder(nn.Module):
    """
    Code Reference: https://www.jianshu.com/p/4dd65ec4654c
    """

    def __init__(self, input_dim, enc_hidden_dim, dec_hidden_dim, num_layers=1, dropout=0.2):
        """
        RNN with batch_first=True and bidirectional=True.
        :param input_dim:
        :param enc_hidden_dim:
        :param dec_hidden_dim:
        :param num_layers:
        :param dropout:
        """
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=enc_hidden_dim, num_layers=num_layers, bidirectional=True,
                          batch_first=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.dropout(src)  # [batch_size, step, dim]
        output, hidden = self.rnn(src)
        """
            outputs = [batch size, step, hidden_dim * num_directions(2)]
            hidden = [num_layers * num_directions, batch size, hidden_dim]
            hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...] of each layer.
            output contains all the hidden from the last layer.

            hidden [-2, :, : ] is the last of the forwards RNN
            hidden [-1, :, : ] is the last of the backwards RNN
        """
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # hidden = [batch size, dec_hidden_dim]
        return output, hidden


class Decoder(nn.Module):
    """
        Code Reference: https://www.jianshu.com/p/4dd65ec4654c
    """

    def __init__(self, enc_hidden_dim, dec_hidden_dim, num_layers=1, dropout=0.2, attention=None, output_dim=1,
                 bidirectional=True):
        """
        RNN with batch_first=True.
        :param enc_hidden_dim:
        :param dec_hidden_dim:
        :param num_layers:
        :param dropout:
        :param attention:
        :param output_dim:
        """
        super(Decoder, self).__init__()
        self.bidirectional = 2 if bidirectional else 1
        if attention is None:
            enc_hidden_dim = 0
        else:
            self.fc = nn.Linear(dec_hidden_dim * self.bidirectional, dec_hidden_dim)
        self.rnn = nn.GRU(input_size=output_dim + enc_hidden_dim * 2, hidden_size=dec_hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dec_hidden_dim * self.bidirectional + output_dim + enc_hidden_dim * 2, output_dim)
        self.attention = attention
        self.num_layers = num_layers

    def forward(self, x, hidden, enc_output=None):
        """
        Process one step.If attention is None or enc_output is not given, then attention is not been used.
        :param x: [batch_size, output_dim]
        :param hidden: [batch_size, dec_hidden_dim] or [num_layers * bidirectional, batch_size, dec_hidden_dim]
        :param enc_output: [batch_size, step, enc_hidden_dim*2]
        :return: [batch_size, output_dim]
        """
        x = self.dropout(x.unsqueeze(1))
        if self.attention is None or enc_output is None:
            if len(hidden.size()) == 2:
                hidden = hidden.repeat(self.num_layers * self.bidirectional, 1, 1)
            # hidden: [num_layers * bidirectional, batch_size, dec_hidden_dim]
            output, hidden = self.rnn(x, hidden)
            x = x.squeeze(1)
            output = output.squeeze(1)
            output = torch.cat((output, x), dim=1)
        else:
            if len(hidden.size()) == 2:
                hidden_attention = hidden
                hidden = hidden.repeat(self.num_layers * self.bidirectional, 1, 1)
            else:
                if self.bidirectional == 1:
                    hidden_attention = hidden[-1, :, :]
                else:
                    hidden_attention = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
            alpha = self.attention(hidden_attention, enc_output).unsqueeze(1)  # [batch_size, 1, step]
            alpha = torch.bmm(alpha, enc_output)  # [batch_size, 1, enc_hidden_dim*2]
            rnn_input = torch.cat((x, alpha), dim=2)  # [batch_size, 1, output_dim+enc_hidden_dim*2]
            # hidden: [num_layers * bidirectional, batch_size, dec_hidden_dim]
            output, hidden = self.rnn(rnn_input, hidden)
            x = x.squeeze(1)
            output = output.squeeze(1)
            alpha = alpha.squeeze(1)
            output = torch.cat((output, alpha, x), dim=1)
        output = self.out(output)  # [batch_size, output_dim]
        return output, hidden


class Seq2Seq(nn.Module):
    """
        Code Reference: https://www.jianshu.com/p/4dd65ec4654c
        Encoder is bidirectional.
    """

    def __init__(self, input_dim, enc_hidden_dim, dec_hidden_dim, enc_num_layers=1, dec_num_layers=1, dropout=0.2,
                 output_dim=1, dec_bidirectional=True, length_pred=24, attention=None):
        """
        Encoder is bidirectional. Decoder can be bidirectional or not bidirectional. RNN is batch_first.
        :param input_dim:
        :param enc_hidden_dim:
        :param dec_hidden_dim:
        :param enc_num_layers:
        :param dec_num_layers:
        :param dropout:
        :param attention: use or not, None.
        :param output_dim: default 1.
        :param dec_bidirectional: default True
        """
        super(Seq2Seq, self).__init__()
        self.length_pred = length_pred
        self.output_dim = output_dim
        self.encoder = Encoder(input_dim=input_dim, enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim,
                               num_layers=enc_num_layers, dropout=dropout)
        self.decoder = Decoder(enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, num_layers=dec_num_layers,
                               dropout=dropout, attention=attention, output_dim=output_dim,
                               bidirectional=dec_bidirectional)

    def forward(self, src, trg=None, start_tag=None, teacher_forcing_ratio=0.5):
        """
        Data should be batch_first.
        :param start_tag:
        :param src:
        :param trg:
        :param teacher_forcing_ratio:
        :return:
        """
        batch_size = src.shape[0]
        result = torch.zeros(batch_size, self.length_pred, self.output_dim, dtype=src.dtype)
        encoder_outputs, hidden = self.encoder(src)
        if start_tag is None or start_tag.shape != torch.Size((batch_size, self.output_dim)):
            start_tag = torch.zeros(batch_size, self.output_dim, dtype=src.dtype)
        if trg is None:
            teacher_forcing_ratio = 0.
        for i in range(self.length_pred):
            output, hidden = self.decoder(start_tag, hidden, encoder_outputs)
            result[:, i, :] = output
            if random.random() < teacher_forcing_ratio:
                start_tag = trg[:, i, :]
            else:
                start_tag = output
        return result


if __name__ == '__main__':
    import numpy as np
    from module.models.self_attention import BahdanauAttention

    a = torch.from_numpy(np.random.uniform(-1., 1., size=(2, 36, 14))).float()
    b = torch.from_numpy(np.random.uniform(-1., 1., size=(2, 24, 1))).float()
    ss = torch.zeros(b.shape[0], b.shape[-1], dtype=a.dtype)
    for _ in range(10):
        enc_hid_dim = pow(2, random.randint(1, 8))
        dec_hid_dim = pow(2, random.randint(1, 8))
        enc_n_layers = random.randint(1, 8)
        dec_n_layers = random.randint(1, 8)
        dec_bid = random.choice([True, False])
        print(enc_hid_dim, dec_hid_dim, enc_n_layers, dec_n_layers, dec_bid)
        attn = BahdanauAttention(enc_hidden_dim=enc_hid_dim, dec_hidden_dim=dec_hid_dim)
        m = Seq2Seq(input_dim=14, enc_hidden_dim=enc_hid_dim, dec_hidden_dim=dec_hid_dim, enc_num_layers=enc_n_layers,
                    dec_num_layers=dec_n_layers, dec_bidirectional=dec_bid, length_pred=24, attention=attn)
        b = m(a)
        print(b.size())
